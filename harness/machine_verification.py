"""Confirm only explicit, fully parsed public requests. Unsupported forms fall back."""
import ast
import csv
import hashlib
import io
import json
import operator
import re
from fractions import Fraction

ARITHMETIC = re.compile(
    r'(?:Compute|Evaluate) (?:the Python integer expression )?(.+?)'
    r'\.(?: Return (?:only the integer|an integer|the integer only)\.)?', re.I)
LOOKUP = re.compile(
    r'Using only ([\w./-]+\.(?:csv|json)), return (\w+) '
    r'(?:where (\w+) equals |for (\w+)=)(.+)\. Return only the value\.', re.I)
INTEGER_EQUALITIES = re.compile(
    r'An integer ([a-zA-Z]) must satisfy \1\s*=\s*([+-]?[0-9]{1,64}) and '
    r'\1\s*=\s*([+-]?[0-9]{1,64}) simultaneously\. Return \1\.')
PAID_REVENUE = re.compile(
    r'(?:Compute|Calculate) (?:the )?(?:total )?paid revenue in base currency as '
    r'quantity\s*\*\s*unit_price\s*\*\s*to_base,\s+using ([\w./-]+\.csv) and ([\w./-]+\.csv)\.\s+'
    r'Include (?:every paid row|all paid rows)\. Exclude (?:only cancelled rows|cancelled rows only)\. '
    r'Missing exchange rates are unknown\. Return (?:only the integer|the total as an integer)\.', re.I)
BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
MISSING = {'', 'na', 'n/a', 'null', 'none', 'unknown', '?'}
UNKNOWN_INPUTS = MISSING | {'??', '????'}


def bounded(value):
    if type(value) is not int or value.bit_length() > 512:
        raise ValueError('integer_limit')
    return value


def integer_expression(expression, *, python=False):
    if len(expression) > 512 or re.fullmatch(r'[0-9+*/%() \t-]+', expression) is None:
        raise ValueError('expression_limit_or_syntax')
    tree = ast.parse(expression, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError('node_limit')

    def compute(node):
        if isinstance(node, ast.Constant):
            return bounded(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = compute(node.operand)
            return bounded(value if isinstance(node.op, ast.UAdd) else -value)
        if not isinstance(node, ast.BinOp):
            raise ValueError('unsupported_expression')
        left, right = compute(node.left), compute(node.right)
        if isinstance(node.op, ast.Pow):
            if not 0 <= right <= 12 or (left == 0 and right == 0 and not python):
                raise ValueError('unsupported_power')
            if left.bit_length() * right > 512:
                raise ValueError('integer_limit')
            return bounded(left ** right)
        if isinstance(node.op, ast.Div):
            if right == 0 or left % right:
                raise ValueError('non_integer_division')
            return bounded(left // right)
        if isinstance(node.op, (ast.FloorDiv, ast.Mod)) and not python:
            raise ValueError('python_operator_requires_explicit_semantics')
        operation = BINARY.get(type(node.op))
        if operation is None:
            raise ValueError('unsupported_operator')
        return bounded(operation(left, right))

    return compute(tree.body)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_json_field')
        result[key] = value
    return result


def csv_records(text):
    if len(text.encode()) > 24000:
        raise ValueError('table_too_large')
    rows = [row for row in csv.reader(io.StringIO(text), strict=True) if row]
    if not rows:
        raise ValueError('empty_table')
    header, *data = rows
    if (len(set(header)) != len(header) or any(not name for name in header)
            or any(len(row) != len(header) for row in data)):
        raise ValueError('invalid_csv_shape')
    return set(header), [dict(zip(header, row)) for row in data]


def lookup_matches(source, text, key_field, key, *, unknown_keys=MISSING):
    if len(text.encode()) > 24000 or type(key) not in (str, int):
        raise ValueError('unsupported_lookup')
    if source.lower().endswith('.csv'):
        _, records = csv_records(text)
        key = str(key)
    else:
        records = json.loads(text, object_pairs_hook=unique_object)
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError('unsupported_json_shape')
    matches = []
    for row in records:
        candidate = row.get(key_field)
        if type(candidate) not in (str, int) or str(candidate).strip().lower() in unknown_keys:
            raise ValueError('unknown_lookup_key')
        if type(candidate) is type(key) and candidate == key:
            matches.append(row)
    return matches


def lookup_field(row, value_field):
    value = row.get(value_field)
    if (type(value) not in (str, int) or str(value).strip().lower() in MISSING
            or str(value) != str(value).strip()):
        raise ValueError('missing_or_unsupported_value')
    return str(value)


def lookup_value(source, text, key_field, key, value_field):
    matches = lookup_matches(source, text, key_field, key)
    if len(matches) != 1:
        raise ValueError('non_unique_lookup')
    return lookup_field(matches[0], value_field)


def check_answer(task_type, instruction, sources, answer):
    """Return a public certificate or None. No file IO, tools or model calls."""
    question = instruction.strip()
    try:
        if task_type == 'math':
            match = ARITHMETIC.fullmatch(question)
            if match is None:
                return None
            expression = match[1]
            expected = str(integer_expression(expression, python='python integer expression' in question.lower()))
            proof = {'kind': 'integer_expression', 'expression': expression}
            involved = ['$instruction']
            equal = bool(re.fullmatch(r'[+-]?[0-9]+', answer)) and int(answer) == int(expected)
        elif task_type == 'search':
            match = LOOKUP.fullmatch(question)
            if match is None:
                return None
            source, field, where_key, for_key, literal = match.groups()
            key = json.loads(literal)
            expected = lookup_value(source, sources[source], where_key or for_key, key, field)
            proof = {'kind': 'unique_lookup', 'source': source, 'key_field': where_key or for_key,
                     'key': key, 'value_field': field}
            involved = ['$instruction', source]
            equal = answer == expected
        else:
            return None
        return {**proof, 'verified': equal, 'answer': expected,
                'source_sha256': {name: hashlib.sha256(sources[name].encode()).hexdigest()
                                  for name in involved}}
    except (ValueError, TypeError, KeyError, SyntaxError, ArithmeticError, RecursionError, csv.Error):
        return None


def decimal_value(value):
    value = value.strip()
    if re.fullmatch(r'[+-]?[0-9]{1,32}(?:\.[0-9]{1,16})?', value) is None:
        raise ValueError('unsupported_decimal')
    return Fraction(value)


def paid_revenue(sales_text, rates_text):
    sales_fields, sales = csv_records(sales_text)
    rate_fields, rates = csv_records(rates_text) if rates_text is not None else ({'currency', 'to_base'}, [])
    if not {'status', 'quantity', 'unit_price', 'currency'} <= sales_fields or not {
            'currency', 'to_base'} <= rate_fields:
        return None
    subtotals = {}
    for row in sales:
        status = row['status'].strip().lower()
        if status == 'cancelled':
            continue
        currency = row['currency'].strip()
        if status != 'paid' or currency.lower() in UNKNOWN_INPUTS:
            return None
        amount = decimal_value(row['quantity']) * decimal_value(row['unit_price'])
        subtotals[currency] = subtotals.get(currency, 0) + amount

    by_currency = {}
    for row in rates:
        currency = row['currency'].strip()
        if currency.lower() in UNKNOWN_INPUTS:
            # A rate without a currency cannot rule out a match.
            return None
        by_currency.setdefault(currency, []).append(row['to_base'].strip())

    total = Fraction(0)
    blocker = None
    for currency, amount in subtotals.items():
        if amount == 0:
            continue
        entries = by_currency.get(currency, [])
        known = [value for value in entries if value.lower() not in UNKNOWN_INPUTS]
        if not known:
            blocker = {'kind': 'missing_exchange_rate', 'currency': currency,
                       'reason': f'No rate is supplied for {currency}, which has a nonzero paid subtotal.'}
            # An integer exchange rate can vary while keeping an integral total.
            if amount.denominator != 1:
                return None
            continue
        choices = {decimal_value(value) for value in known}
        if len(known) != len(entries) or len(choices) != 1:
            return None
        total += amount * choices.pop()
    if total.denominator != 1:
        return None
    return blocker or {'kind': 'paid_revenue', 'answer': str(total.numerator)}


def resolve_public_task(task_type, instruction, read_source):
    """Resolve only complete supported requests using their named original inputs."""
    question = instruction.strip()
    try:
        if task_type == 'math':
            equalities = INTEGER_EQUALITIES.fullmatch(question)
            if equalities:
                variable, left, right = equalities.groups()
                if int(left) == int(right):
                    return {'kind': 'integer_equalities', 'answer': str(int(left))}
                return {'kind': 'contradictory_equalities',
                        'reason': f'The integer {variable} must equal both {left} and {right}.'}
            expression = ARITHMETIC.fullmatch(question)
            if expression:
                python = 'python integer expression' in question.lower()
                # Python true division introduces floating-point semantics.
                if not python or '/' not in expression[1].replace('//', ''):
                    return {'kind': 'integer_expression',
                            'answer': str(integer_expression(expression[1], python=python))}
        elif task_type == 'search':
            lookup = LOOKUP.fullmatch(question)
            if lookup:
                source, field, where_key, for_key, literal = lookup.groups()
                key = json.loads(literal)
                if source.lower().endswith('.csv') and not isinstance(key, str):
                    return None  # CSV has no field type for comparing numeric keys.
                matches = lookup_matches(source, read_source(source), where_key or for_key, key,
                                         unknown_keys=UNKNOWN_INPUTS)
                values = {lookup_field(row, field) for row in matches}
                if any(value.lower() in UNKNOWN_INPUTS for value in values):
                    return None
                if len(values) == 1:
                    return {'kind': 'lookup', 'answer': values.pop()}
                if len(values) > 1:
                    return {'kind': 'ambiguous_lookup', 'source': source, 'values': sorted(values),
                            'reason': f'Matching records in {source} give different values of {field}.'}
        elif task_type == 'data_analysis':
            revenue = PAID_REVENUE.fullmatch(question)
            if revenue and revenue[1] != revenue[2]:
                sales_text = read_source(revenue[1])
                try:
                    rates_text = read_source(revenue[2])
                except FileNotFoundError:
                    rates_text = None
                return paid_revenue(sales_text, rates_text)
    except (ValueError, TypeError, KeyError, SyntaxError, ArithmeticError, RecursionError, csv.Error, OSError):
        return None
    return None
