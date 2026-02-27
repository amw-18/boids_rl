import argparse
import sys
import sqlite3
import subprocess
from pathlib import Path

# Add parent directory to path to import db script
sys.path.append(str(Path(__file__).parent))
from db import init_db, get_db

def queue_run(commit_hash, name):
    conn = init_db()
    # verify commit exists
    try:
        if commit_hash.lower() != "head":
            subprocess.check_output(["git", "rev-parse", "--verify", f"{commit_hash}^{{commit}}"])
        else:
            commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except subprocess.CalledProcessError:
        print(f"Error: Commit '{commit_hash}' not found.")
        sys.exit(1)
        
    cursor = conn.cursor()
    cursor.execute("INSERT INTO runs (commit_hash, name, status) VALUES (?, ?, 'pending')", 
                   (commit_hash, name))
    conn.commit()
    print(f"Queued run #{cursor.lastrowid} for commit {commit_hash[:8]} ({name})")

def list_runs():
    init_db()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, commit_hash, name, status, created_at FROM runs ORDER BY id DESC")
    rows = cursor.fetchall()
    
    print(f"{'ID':<5} | {'Commit':<10} | {'Status':<12} | {'Name':<30} | {'Created At'}")
    print("-" * 80)
    for r in rows:
        commit_short = r['commit_hash'][:8]
        print(f"{r['id']:<5} | {commit_short:<10} | {r['status']:<12} | {r['name']:<30} | {r['created_at']}")

def cancel_run(run_id):
    init_db()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT status, pid, worktree_dir FROM runs WHERE id = ?", (run_id,))
    run = cursor.fetchone()
    if not run:
        print(f"Run {run_id} not found.")
        return
        
    if run['status'] == 'pending':
        cursor.execute("UPDATE runs SET status = 'cancelled' WHERE id = ?", (run_id,))
        conn.commit()
        print(f"Run {run_id} cancelled.")
    elif run['status'] == 'active':
        if run['pid']:
            import os
            import signal
            try:
                os.kill(run['pid'], signal.SIGTERM)
                print(f"Sent SIGTERM to process {run['pid']}")
            except ProcessLookupError:
                print("Process already terminated.")
        cursor.execute("UPDATE runs SET status = 'cancelled' WHERE id = ?", (run_id,))
        conn.commit()
        print(f"Active Run {run_id} cancelled. The worker will clean up the worktree shortly.")
    else:
        print(f"Cannot cancel run {run_id} (current status: {run['status']})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage local RL training runs")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Queue
    p_queue = subparsers.add_parser("queue", help="Queue a new run")
    p_queue.add_argument("commit_hash", help="Git commit hash to run (or HEAD)")
    p_queue.add_argument("--name", required=True, help="Experiment name")
    
    # List
    p_list = subparsers.add_parser("list", help="List all runs")
    
    # Cancel
    p_cancel = subparsers.add_parser("cancel", help="Cancel a run")
    p_cancel.add_argument("run_id", type=int, help="Run ID to cancel")
    
    args = parser.parse_args()
    
    if args.command == "queue":
        queue_run(args.commit_hash, args.name)
    elif args.command == "list":
        list_runs()
    elif args.command == "cancel":
        cancel_run(args.run_id)
