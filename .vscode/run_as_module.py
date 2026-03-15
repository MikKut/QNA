# .vscode/run_as_module.py
# Запускає активний файл як "python -m <pkg.path>"
import os, sys, runpy, pathlib

root = pathlib.Path(os.environ.get("WORKSPACE_FOLDER", os.getcwd())).resolve()
file_path = pathlib.Path(sys.argv[1]).resolve()

# Перетворюємо шлях файлу на ім'я модуля: Code/preprocess/x.py -> Code.preprocess.x
rel = file_path.relative_to(root)
mod = ".".join(rel.with_suffix("").parts)
mod = mod.replace(".__init__", "")

# Пробрасываем оригінальні CLI-аргументи (після шляху файлу)
sys.argv = [mod] + sys.argv[2:]
runpy.run_module(mod, run_name="__main__")
