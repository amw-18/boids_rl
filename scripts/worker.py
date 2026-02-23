import os
import sys
import time
import subprocess
import shutil
import uuid
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from db import init_db, get_db

class Worker:
    def __init__(self):
        self.conn = init_db()
        
    def poll(self):
        while True:
            cursor = self.conn.cursor()
            # Find next pending run
            cursor.execute("SELECT * FROM runs WHERE status = 'pending' ORDER BY id ASC LIMIT 1")
            run = cursor.fetchone()
            
            if run:
                try:
                    self.execute_run(run)
                except Exception as e:
                    print(f"Error executing run {run['id']}: {e}")
                    cursor.execute("UPDATE runs SET status = 'failed' WHERE id = ?", (run['id'],))
                    self.conn.commit()
            else:
                time.sleep(5)
                
    def execute_run(self, run):
        run_id = run['id']
        commit_hash = run['commit_hash']
        name = run['name']
        
        print(f"\n[{datetime.now()}] Starting Run #{run_id} ({name}) @ {commit_hash[:8]}")
        
        pid = os.getpid()
        cursor = self.conn.cursor()
        
        # Mark as active
        worktree_dir = os.path.abspath(f".runs/run_{run_id}")
        wandb_run_id = str(uuid.uuid4().hex)[:8] # standard 8 char wandb id
        
        cursor.execute('''
            UPDATE runs SET status = 'active', started_at = CURRENT_TIMESTAMP, 
            pid = ?, worktree_dir = ? WHERE id = ?
        ''', (pid, worktree_dir, run_id))
        self.conn.commit()
        
        # 1. Setup Worktree
        os.makedirs(".runs", exist_ok=True)
        if os.path.exists(worktree_dir):
            shutil.rmtree(worktree_dir)
            
        print(f"Creating Git Worktree at {worktree_dir}")
        subprocess.check_call(["git", "worktree", "add", "--detach", worktree_dir, commit_hash])
        
        try:
            # 2. Execute Training
            env = os.environ.copy()
            env["WANDB_RUN_ID"] = wandb_run_id
            env["WANDB_RUN_GROUP"] = "queue"
            env["WANDB_NAME"] = name
            
            print(f"Launching training script... (WANDB_RUN_ID={wandb_run_id})")
            
            # Use `uv run` to ensure correct environment
            cmd = ["uv", "run", "python", "src/murmur_rl/training/runner.py"]
            
            process = subprocess.Popen(
                cmd,
                cwd=worktree_dir,
                env=env
            )
            
            # Periodically poll to see if the run was cancelled while running
            while process.poll() is None:
                cursor.execute("SELECT status FROM runs WHERE id = ?", (run_id,))
                current_status = cursor.fetchone()['status']
                if current_status == 'cancelled':
                    print("Run was cancelled from the database. Terminating process...")
                    process.terminate()
                    process.wait()
                    break
                time.sleep(2)
            
            if process.returncode == 0:
                cursor.execute("UPDATE runs SET status = 'completed', completed_at = CURRENT_TIMESTAMP WHERE id = ?", (run_id,))
                print(f"Run #{run_id} completed successfully.")
            else:
                cursor.execute("SELECT status FROM runs WHERE id = ?", (run_id,))
                current_status = cursor.fetchone()['status']
                if current_status != 'cancelled':
                    cursor.execute("UPDATE runs SET status = 'failed', completed_at = CURRENT_TIMESTAMP WHERE id = ?", (run_id,))
                    print(f"Run #{run_id} failed with return code {process.returncode}.")
                    
        finally:
            self.conn.commit()
            # 3. Cleanup Worktree
            print(f"Cleaning up worktree at {worktree_dir}")
            try:
                subprocess.check_call(["git", "worktree", "remove", "--force", worktree_dir])
            except subprocess.CalledProcessError:
                shutil.rmtree(worktree_dir, ignore_errors=True)
                subprocess.check_call(["git", "worktree", "prune"])
            print("-" * 40)

if __name__ == "__main__":
    print("Starting Worker Daemon. Watching queue...")
    worker = Worker()
    worker.poll()
