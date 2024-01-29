import pkg_resources
import ast

script_path = 'detect.py'

with open(script_path, 'r') as file:
    script_content = file.read()

parsed_script = ast.parse(script_content)
imports = [node.name for node in ast.walk(parsed_script) if isinstance(node, ast.Import)]
import_froms = [node.module for node in ast.walk(parsed_script) if isinstance(node, ast.ImportFrom)]
all_imports = set(imports + import_froms)

installed_packages = {pkg.key for pkg in pkg_resources.working_set}

requirements = []
for imp in all_imports:
    base_module = imp.split('.')[0]
    if base_module in installed_packages:
        version = pkg_resources.get_distribution(base_module).version
        requirements.append(f"{base_module}=={version}")

with open('requirements.txt', 'w') as file:
    for req in sorted(set(requirements)):
        file.write(req + '\n')

print("requirements.txt generated.")