from codebleu import calc_codebleu

y_pred = [
    """for (int i = 0; i < 10; i++) {
    printf("%d", i*i);
    }"""
    ]
y_true = [
    """for (int i = 0; i < 10; i++) {
    printf("%d", i);
    }"""
    ]

prediction = "def add ( a , b ) :\n return a + b"
reference = "def sum ( first , second ) :\n return second + first"

result = calc_codebleu([reference], [prediction], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
print(result)

print(calc_codebleu(y_pred, y_true, lang="cpp", weights=[0.15, 0.15, 0.30, 0.40]))