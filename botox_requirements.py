with open('requirements.txt', 'r') as file:
    lines = file.readlines()

new_lines = []
for line in lines:
    parts = line.split('=')
    if len(parts) >= 2:
        new_line = parts[0] + '==' + parts[1] + '\n'
        new_lines.append(new_line)

with open('requirements.txt', 'w') as file:
    file.writelines(new_lines)
