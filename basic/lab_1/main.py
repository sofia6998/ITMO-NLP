import re
from nltk.stem.snowball import SnowballStemmer
import pymorphy2

text = "Привет я Соня))  Я тут. Как дела :p a = b + c Я живу по адресу Ул. Большая д. 5 к. 3 кв. 34. что делать:( a = b*c = (c+d)^2  q +a = b +c  " \
       " (a:p) +(a+d+c)*2=2a. Мой номер телефона 8921-123-23-23. А номер моей сестры 8(821)-123-23-24"

text1 = " (a+b) +(a+d+c)*2=2a"
address_short = r'(ул|д|к|кв)\.'
smiles = r'[:;%BВ]([\(\)]+|[\*Pp0oOоОDd\\\/#}{])|[\(\)]{2,}'
math_variable = '(([0-9]*[A-Za-z])|(-?[0-9]+))(\s?)'
math_operator = '([+\-\*\/:%]|(\^))(\s?)'
math_multi_opers_plus_vars = "(" + math_operator + math_variable + ")*"
math_eq_unit = "((-?)" + math_variable + math_multi_opers_plus_vars + ")"
math_eq_unit_with_brackets = "((\((\s?)" + math_eq_unit + "\)(\s?))|" + math_eq_unit + ")"
math_eq_part = "((-?)" + math_eq_unit_with_brackets + "(" + math_operator + math_eq_unit_with_brackets + ")*)"
math_equations_str = "(" + math_eq_part + ")" + "(=(\s?)(-?)" + math_eq_part + ")"
math_equations = r'%s' % math_equations_str

phone_start = '\+?\d(-?\((\d)+\))?' # +7-(812)
digits_with_brackets = "((\((\s?)" + '(\d)+' + "\)(\s?))|" + '(\d)+' + ")"
phone_number_str = phone_start + "(-?" + digits_with_brackets + ")+"
phone_number = r'%s' % phone_number_str

print(phone_number_str)
print(math_equations_str)

#        PART 1


def address_short_match(source):
    match = re.search(address_short, source, re.IGNORECASE)
    return match


def math_match(source):
    match = re.search(math_equations, source)
    return match


def emoji_match(source):
    match = re.search(smiles, source)
    return match


def phone_match(source):
    match = re.search(phone_number, source)
    return match


def split_and_tokenize(source, match):
    [a, b] = source.split(match.group(0), 1)
    return [*tokenize(a),  match.group(0).strip(), *tokenize(b)]


def tokenize(in_source):
    source = in_source.strip()

    match = address_short_match(source)
    if match:
        return split_and_tokenize(source, match)

    match = math_match(source)
    if match:
        return split_and_tokenize(source, match)

    match = emoji_match(source)
    if match:
        return split_and_tokenize(source, match)

    match = phone_match(source)
    if match:
        return split_and_tokenize(source, match)

    return [*re.split('[^a-zA-Z0-9А-Яа-я]', source)]

#        PART 2


def get_stems(tokens_array):
    stemmer = SnowballStemmer("russian")
    res = []
    for t in tokens_array:
        stem = stemmer.stem(t)
        res.append(stem)

    return res


def get_lems(tokens_array):
    morph = pymorphy2.MorphAnalyzer(lang='ru')
    res = []
    for t in tokens_array:
        p = morph.parse(t)[0]
        res.append(p.normal_form)

    return res



if __name__ == '__main__':
    print(tokenize(text))
    dirty_tokens = tokenize(text)
    tokens = list(filter(lambda x: len(x) > 0, dirty_tokens))
    print(tokens)
    print(len(tokens))

    stems = get_stems(tokens)
    print(stems)
    lems = get_lems(tokens)
    print(lems)
