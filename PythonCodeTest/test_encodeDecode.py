import os

# --- String-level encode/decode ---

def encodeString(stringVal):
    encodedList = []
    prevChar = None
    count = 0
    for char in stringVal:
        if prevChar != char and prevChar is not None:
            encodedList.append((prevChar, count))
            count = 0
        prevChar = char
        count += 1
    encodedList.append((prevChar, count))
    return encodedList

def decodeString(encodedList):
    return ''.join(char * count for char, count in encodedList)


# --- File-level encode/decode ---

def encodeFile(filename, newFilename):
    with open(filename, 'r') as f:
        content = f.read()
    encodedList = encodeString(content)
    with open(newFilename, 'w') as f:
        for char, count in encodedList:
            f.write(f'{count},{ord(char)}\n')

def decodeFile(filename):
    encodedList = []
    with open(filename, 'r') as f:
        for line in f:
            count, char_ord = line.strip().split(',')
            encodedList.append((chr(int(char_ord)), int(count)))
    decoded = decodeString(encodedList)
    decoded_filename = filename.replace('_encoded', '_decoded')
    with open(decoded_filename, 'w') as f:
        f.write(decoded)
    print(f'Decoded file path: {decoded_filename}')
    print(f'Decoded file size: {os.path.getsize(decoded_filename)}')


# --- Main ---

if __name__ == '__main__':
    file_path = os.path.join(os.path.dirname(__file__), 'datasets', 'sampleText.txt')
    encoded_path = file_path.replace('.txt', '_encoded.txt')

    print(f'Original file size: {os.path.getsize(file_path)}')

    encodeFile(file_path, encoded_path)
    print(f'Encoded file path: {encoded_path}')
    print(f'Encoded file size: {os.path.getsize(encoded_path)}')

    decodeFile(encoded_path)
