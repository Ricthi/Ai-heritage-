import os
import sys

# Define the absolute path to main_app.py
current_dir = os.path.dirname(os.path.abspath(__file__))
main_app_path = os.path.join(current_dir, 'tamil_heritage_ai', 'Model-Creation', 'main_app.py')

# Add the Model-Creation directory to sys.path so it can import its local files (like tamil_charset)
sys.path.insert(0, os.path.dirname(main_app_path))

# Execute main_app.py within the correct context so __file__ points to main_app.py
with open(main_app_path, encoding='utf-8') as f:
    code = compile(f.read(), main_app_path, 'exec')
    # We pass __file__ as main_app_path so that all os.path.dirname(__file__) calls inside main_app.py work correctly!
    exec(code, {'__file__': main_app_path, '__name__': '__main__'})
