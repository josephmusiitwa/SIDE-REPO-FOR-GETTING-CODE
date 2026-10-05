import os
import glob
import re

directories = [
    r"c:\Users\josep\OneDrive\Desktop\SIDE-REPO-FOR-GETTING-CODE\meanshift23\examples",
    r"c:\Users\josep\OneDrive\Desktop\SIDE-REPO-FOR-GETTING-CODE\meanshift23\include\meanshift",
    r"c:\Users\josep\OneDrive\Desktop\SIDE-REPO-FOR-GETTING-CODE\meanshift23\src",
    r"c:\Users\josep\OneDrive\Desktop\SIDE-REPO-FOR-GETTING-CODE\meanshift23\tests"
]

for d in directories:
    for ext in ("*.cpp", "*.hpp"):
        for filepath in glob.glob(os.path.join(d, ext)):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if it already has 'using namespace std;'
            if 'using namespace std;' not in content:
                # Find the last include
                lines = content.split('\n')
                last_include_idx = -1
                for i, line in enumerate(lines):
                    if line.startswith('#include'):
                        last_include_idx = i
                
                if last_include_idx != -1:
                    lines.insert(last_include_idx + 1, '\nusing namespace std;')
                else:
                    # if no include, put it at the top (after header guards if present)
                    insert_idx = 0
                    for i, line in enumerate(lines):
                        if not line.startswith('#ifndef') and not line.startswith('#define'):
                            insert_idx = i
                            break
                    lines.insert(insert_idx, '\nusing namespace std;')
                
                content = '\n'.join(lines)
            
            # Remove std::
            content = content.replace('std::', '')
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Updated {filepath}")
