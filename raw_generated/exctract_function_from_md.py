import sys
from pyparseit import parse_markdown_file

file_path = sys.argv[1]
language = 'c'

output_file = sys.argv[2]

snippets = parse_markdown_file(file_path, language=language)



#append to output file
with open(output_file, 'a') as f:
    for snippet in snippets:
        cleaned = snippet.content.strip('#include <immintrin.h>')
        print(f"Language: {snippet.language}\nContent:\n{cleaned}\n")
        f.write(f"{cleaned}")
        f.write("\n")