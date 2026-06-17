import os 
import subprocess

def save_git_state(save_dir):
    """
    Save the current git state including commit hash and uncommitted changes.
    
    Args:
        save_dir: Directory to save git state information
    """
    try:
        # Check if we're in a git repository
        result = subprocess.run(
            ['git', 'rev-parse', '--git-dir'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Get current commit hash
        hash_result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        current_hash = hash_result.stdout.strip()
        
        # Get uncommitted changes (staged + unstaged)
        diff_result = subprocess.run(
            ['git', 'diff', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        uncommitted_changes = diff_result.stdout
        
        # Get untracked files
        untracked_result = subprocess.run(
            ['git', 'ls-files', '--others', '--exclude-standard'],
            capture_output=True,
            text=True,
            check=True
        )
        untracked_files = untracked_result.stdout.strip().split('\n') if untracked_result.stdout.strip() else []
        
        # Save commit hash
        hash_file = os.path.join(save_dir, 'git_commit_hash.txt')
        with open(hash_file, 'w') as f:
            f.write(current_hash + '\n')
        
        # Save uncommitted changes (tracked files) if any exist
        if uncommitted_changes.strip():
            patch_file = os.path.join(save_dir, f'uncommitted_changes_{current_hash[:8]}.patch')
            with open(patch_file, 'w') as f:
                f.write(uncommitted_changes)
            print(f"Saved uncommitted changes to: {patch_file}")
            print(f"  Apply with: git apply {patch_file}")
        
        # Save untracked files — include contents for small files (<400KB),
        # list name only for larger files (reading large/binary files can
        # block for minutes and cause DDP NCCL timeouts)
        if untracked_files:
            untracked_file_path = os.path.join(save_dir, f'untracked_files_{current_hash[:8]}.txt')
            max_content_size = 400 * 1024  # 400 KB
            with open(untracked_file_path, 'w') as f:
                f.write("# Untracked files at training time\n")
                f.write(f"# Commit: {current_hash}\n\n")
                for untracked_file in untracked_files:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"# File: {untracked_file}\n")
                    f.write(f"{'='*80}\n")
                    try:
                        file_size = os.path.getsize(untracked_file)
                        if file_size <= max_content_size:
                            with open(untracked_file, 'r', encoding='utf-8', errors='ignore') as uf:
                                file_content = uf.read()
                            f.write(file_content)
                            if not file_content.endswith('\n'):
                                f.write('\n')
                        else:
                            f.write(f"# Skipped: {file_size / 1024:.0f} KB (>{max_content_size // 1024} KB limit)\n")
                    except Exception as e:
                        f.write(f"# Error reading file: {e}\n")
            print(f"Saved {len(untracked_files)} untracked file(s) to: {untracked_file_path}")
        
        if not uncommitted_changes.strip() and not untracked_files:
            print("No uncommitted changes or untracked files detected.")
        
        print(f"Git commit hash: {current_hash}")
        
    except subprocess.CalledProcessError:
        print("Warning: Not in a git repository or git command failed. Skipping git state save.")
    except FileNotFoundError:
        print("Warning: Git is not installed. Skipping git state save.")
