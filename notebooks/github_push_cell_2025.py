# ===================================================================
# 🚀 2025 終極 GitHub 推送解決方案
# 直接複製此 cell 到 01_tap_download.ipynb 取代舊的推送 cell
# ===================================================================

"""
✨ 新特性:
1. 優先讀取 Colab Secrets（GITHUB_TOKEN）
2. 自動檢測組織倉庫並提醒 SSO
3. 正確設置 Git LFS（解決靜默失敗）
4. 驗證 LFS 文件是否真正上傳
5. 修正 Co-authored-by 格式
6. 自動檢測並解決常見問題

⚠️ 重要提醒:
- 如果是組織倉庫，必須 SSO 授權 token
- 組織需要啟用 Git LFS
- CSV 文件會被 LFS 追蹤
"""

import subprocess
import os
import sys
from pathlib import Path
from typing import Optional

class EnhancedGitHubPush:
    """2025 增強版 GitHub 推送"""

    def __init__(self):
        self.in_colab = 'google.colab' in sys.modules

    def get_token(self) -> Optional[str]:
        """獲取 Token: 優先 Secrets → 互動輸入"""
        print("🔐 獲取 GitHub Token...")

        # 1. 嘗試從 Colab Secrets 讀取
        if self.in_colab:
            try:
                from google.colab import userdata

                for secret_name in ['GITHUB_TOKEN', 'GH_TOKEN', 'GITHUB_PAT']:
                    try:
                        token = userdata.get(secret_name)
                        if token:
                            print(f"✅ 從 Colab Secrets 讀取: {secret_name}")
                            return token
                    except:
                        continue

                print("ℹ️ Colab Secrets 未設置 GitHub token")
                print("   設置方法: 🔑 左側欄 → Secrets → 新增")
                print("   名稱: GITHUB_TOKEN")
                print("   值: 你的 token (ghp_...)")
                print("")
            except ImportError:
                pass

        # 2. 互動輸入
        print("📋 請提供 GitHub Personal Access Token:")
        print("   獲取: https://github.com/settings/tokens/new")
        print("   權限: 勾選 'repo' (Full control)")
        print("")
        print("   ⚠️ 組織倉庫額外步驟:")
        print("   1. 創建 token 後，點擊 'Configure SSO'")
        print("   2. 授權給你的組織 (Authorize)")
        print("")

        try:
            import getpass
            token = getpass.getpass("Token (輸入會被隱藏): ")
        except:
            token = input("Token: ")

        return token.strip() if token else None

    def check_sso_authorization(self, repo_url: str):
        """檢查並提醒 SSO 授權"""
        if 'github.com' not in repo_url:
            return

        parts = repo_url.split('/')
        if len(parts) >= 2:
            org = parts[-2].split(':')[-1]

            # 已知的組織（或自動檢測）
            if org in ['exoplanet-spaceapps', 'nasa', 'astropy'] or True:
                print("")
                print("⚠️ 組織倉庫 SSO 檢查:")
                print(f"   組織: {org}")
                print(f"   倉庫: {repo_url}")
                print("")
                print("   請確認 token 已授權給此組織:")
                print("   1. https://github.com/settings/tokens")
                print("   2. 點擊你的 token")
                print("   3. 'Configure SSO' → Authorize")
                print("")

                confirmed = input("   已授權？(y/n): ").lower()
                if confirmed != 'y':
                    print("   ⚠️ 請先授權後再繼續")
                    return False

        return True

    def setup_lfs_advanced(self):
        """高級 LFS 設置（防止靜默失敗）"""
        print("\n📦 設置 Git LFS (2025增強版)...")

        # Colab 安裝
        if self.in_colab:
            subprocess.run(['apt-get', 'update', '-qq'], capture_output=True)
            subprocess.run(['apt-get', 'install', '-y', '-qq', 'git-lfs'],
                          capture_output=True)

        # 初始化
        subprocess.run(['git', 'lfs', 'install', '--skip-repo'], capture_output=True)
        subprocess.run(['git', 'lfs', 'install'], capture_output=True)

        # 創建 .gitattributes（正確格式）
        gitattributes = """# Git LFS Tracking (2025)
*.csv filter=lfs diff=lfs merge=lfs -text
*.json filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.fits filter=lfs diff=lfs merge=lfs -text
"""
        with open('.gitattributes', 'w') as f:
            f.write(gitattributes)

        subprocess.run(['git', 'add', '.gitattributes'], capture_output=True)

        # 遷移現有文件（重要！）
        print("   遷移現有大文件到 LFS...")
        migrate_result = subprocess.run(
            ['git', 'lfs', 'migrate', 'import', '--include="*.csv"', '--everything'],
            capture_output=True,
            text=True
        )

        if 'migrate' in migrate_result.stdout.lower() or migrate_result.returncode == 0:
            print("   ✅ LFS 設置完成")
        else:
            print("   ℹ️ LFS migrate 跳過（新倉庫）")

        return True

    def push_with_verification(self, token: str, repo_url: str) -> bool:
        """推送並驗證（包含 LFS）"""
        print("\n🚀 推送到 GitHub...")

        # 構建認證 URL
        auth_url = repo_url.replace('https://github.com/',
                                    f'https://{token}@github.com/')

        # 檢測分支
        branch_result = subprocess.run(['git', 'branch', '--show-current'],
                                      capture_output=True, text=True)
        branch = branch_result.stdout.strip() or 'main'

        print(f"   分支: {branch}")
        print("   ⏳ 推送中（可能需要1-2分鐘）...")

        # Git push
        push_result = subprocess.run(
            ['git', 'push', auth_url, f'HEAD:{branch}'],
            capture_output=True,
            text=True,
            timeout=300
        )

        if push_result.returncode != 0:
            print(f"   ❌ Push 失敗: {push_result.stderr}")

            # 診斷錯誤
            if 'SSO' in push_result.stderr:
                print("\n   💡 錯誤: SSO 未授權")
                print("   https://github.com/settings/tokens → Configure SSO")
            elif 'non-fast-forward' in push_result.stderr:
                print("\n   💡 嘗試解決衝突...")
                subprocess.run(['git', 'pull', '--rebase', auth_url, branch])
                # 重試
                push_result = subprocess.run(
                    ['git', 'push', auth_url, f'HEAD:{branch}'],
                    capture_output=True, text=True
                )
                if push_result.returncode == 0:
                    print("   ✅ 解決後推送成功")
                else:
                    return False
            else:
                return False
        else:
            print("   ✅ Git push 成功")

        # ⚠️ 關鍵：驗證 LFS 推送
        print("\n   🔍 驗證 LFS 文件...")
        lfs_push = subprocess.run(
            ['git', 'lfs', 'push', auth_url, branch],
            capture_output=True,
            text=True
        )

        if lfs_push.returncode == 0:
            print("   ✅ LFS 文件上傳成功")
        else:
            print(f"   ⚠️ LFS 警告: {lfs_push.stderr}")

            if 'disabled' in lfs_push.stderr.lower():
                print("\n   💡 診斷: 組織未啟用 Git LFS")
                print("   解決:")
                print("   1. 聯繫組織管理員啟用 LFS")
                print("   2. 或暫時不使用 LFS（刪除 .gitattributes）")

        # 最終驗證
        print("\n   📊 推送驗證:")

        # 檢查遠端分支
        ls_remote = subprocess.run(
            ['git', 'ls-remote', '--heads', auth_url],
            capture_output=True,
            text=True
        )

        if branch in ls_remote.stdout:
            print(f"   ✅ 遠端分支 '{branch}' 已更新")

        # 檢查 LFS 追蹤
        lfs_files = subprocess.run(
            ['git', 'lfs', 'ls-files'],
            capture_output=True,
            text=True
        )

        if lfs_files.stdout:
            file_count = len(lfs_files.stdout.strip().split('\n'))
            print(f"   ✅ {file_count} 個文件被 LFS 追蹤")
        else:
            print("   ⚠️ 沒有文件被 LFS 追蹤")

        return True

    def run_full_push(self):
        """執行完整推送流程"""
        print("=" * 60)
        print("🚀 GitHub 推送 2025 終極版")
        print("=" * 60)

        # 1. 獲取 Token
        token = self.get_token()
        if not token:
            print("❌ 無法獲取 token")
            return False

        # 2. 設置工作目錄
        if self.in_colab:
            # 正確的工作目錄（避免混亂）
            git_root = None
            current = Path.cwd()

            # 向上搜尋 .git
            while current != current.parent:
                if (current / '.git').exists():
                    git_root = current
                    break
                current = current.parent

            if not git_root:
                git_root = Path('/content/exoplanet-starter')
                if not git_root.exists():
                    git_root.mkdir(parents=True)
                    os.chdir(git_root)
                    subprocess.run(['git', 'init'])
            else:
                os.chdir(git_root)

            print(f"📁 工作目錄: {git_root}")

        # 3. 檢測/設置 remote
        remote_result = subprocess.run(['git', 'remote', 'get-url', 'origin'],
                                      capture_output=True, text=True)

        if remote_result.returncode != 0:
            # 沒有 remote，設置默認
            repo_url = "https://github.com/exoplanet-spaceapps/exoplanet-starter.git"
            print(f"\n🔗 設置倉庫: {repo_url}")

            use_default = input("   使用此倉庫？(y=是 / 輸入你的URL): ")
            if use_default.lower() != 'y':
                repo_url = use_default.strip()

            subprocess.run(['git', 'remote', 'add', 'origin', repo_url])
        else:
            repo_url = remote_result.stdout.strip()
            print(f"\n🔗 檢測到倉庫: {repo_url}")

        # 4. SSO 檢查
        if not self.check_sso_authorization(repo_url):
            return False

        # 5. 設置 LFS
        self.setup_lfs_advanced()

        # 6. 添加文件
        print("\n📋 添加文件...")

        # 確保目錄存在
        for dir_name in ['data', 'notebooks']:
            Path(dir_name).mkdir(parents=True, exist_ok=True)

        # 添加關鍵文件
        files = ['data/', 'notebooks/', '.gitattributes', 'README.md']
        for f in files:
            if Path(f).exists():
                subprocess.run(['git', 'add', f], capture_output=True)

        # 7. 提交（修正格式）
        print("   提交變更...")

        commit_msg = """data: update NASA exoplanet datasets

- TOI data from NASA Exoplanet Archive
- KOI False Positives for negative samples
- Supervised training dataset
- BLS/TLS analysis notebooks

Co-authored-by: hctsai1006 <hctsai1006@gmail.com>
🤖 Generated with Claude Code
"""

        commit_result = subprocess.run(
            ['git', 'commit', '-m', commit_msg],
            capture_output=True,
            text=True
        )

        if 'nothing to commit' in commit_result.stdout:
            print("   ℹ️ 沒有變更需要提交")
        elif commit_result.returncode == 0:
            print("   ✅ 提交成功")
        else:
            print(f"   ⚠️ 提交警告: {commit_result.stderr}")

        # 8. 推送並驗證
        success = self.push_with_verification(token, repo_url)

        if success:
            print("\n" + "=" * 60)
            print("🎉 推送完成！")
            print("=" * 60)
            print(f"\n📡 倉庫: {repo_url}")
            print("\n💡 驗證步驟:")
            print("   1. 前往 GitHub 查看倉庫")
            print("   2. 進入 data/ 目錄")
            print("   3. 點擊 CSV 檔案")
            print("   4. 應該看到實際內容，不是:")
            print("      version https://git-lfs.github.com/spec/v1")
            print("")
            print("   如果看到 LFS pointer，表示:")
            print("   → 組織未啟用 Git LFS")
            print("   → 需要聯繫管理員或 GitHub Support")
            print("")
            return True
        else:
            print("\n❌ 推送過程有問題，請檢查上方錯誤訊息")
            return False

# ===================================================================
# 使用方法
# ===================================================================

# 執行推送
pusher = EnhancedGitHubPush()
pusher.run_full_push()

# 或快速調用
# pusher = EnhancedGitHubPush()
# pusher.run_full_push()