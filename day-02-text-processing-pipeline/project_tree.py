import os

def generate_selective_tree(startpath, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        # Project root
        f.write(f"{os.path.basename(startpath)}/\n")
        
        # Top-level directories and files
        top_level = ['src', 'data', 'notebooks', 'tests', 'app.py', 'requirements.txt']
        
        for item in top_level:
            full_path = os.path.join(startpath, item)
            
            if os.path.exists(full_path):
                if os.path.isdir(full_path):
                    f.write(f"├── {item}/\n")
                    
                    # For src directory, show immediate subdirectories
                    if item == 'src':
                        for subdir in os.listdir(full_path):
                            subdir_path = os.path.join(full_path, subdir)
                            if os.path.isdir(subdir_path):
                                f.write(f"│   ├── {subdir}/\n")
                                
                                # Show files in preprocessing
                                if subdir == 'preprocessing':
                                    for file in os.listdir(subdir_path):
                                        if file.endswith('.py'):
                                            f.write(f"│   │   ├── {file}\n")
                else:
                    f.write(f"├── {item}\n")

# Specify the path and output file
project_path = r"D:\myRepo\llm-python-learning-journey\day-02-text-processing-pipeline"
output_path = os.path.join(project_path, "project_structure.txt")

# Generate the tree
generate_selective_tree(project_path, output_path)

print(f"Selective directory tree has been saved to {output_path}")

# Read and print the contents of the file
with open(output_path, 'r', encoding='utf-8') as f:
    print(f.read())