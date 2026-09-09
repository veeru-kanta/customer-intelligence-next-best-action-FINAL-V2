
import subprocess, sys
steps=[
    ["src/preprocess.py"],
    ["src/statistical_tests.py"],
    ["src/churn_model.py"],
    ["src/nlp_text_mining.py"],
    ["src/next_best_action.py"],
]
for s in steps:
    print("\n>>>",s[0])
    subprocess.run([sys.executable]+s,check=True)
print("\nPIPELINE COMPLETE")
