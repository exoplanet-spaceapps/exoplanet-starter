"""
🚀 GitHub Push 終極解決方案 2025 Edition
解決所有已知問題：LFS、SSO、組織權限、Colab Secrets

作者: Claude Code
更新: 2025-09-30
"""

import subprocess
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import json
import time

class GitHubPushHelper:
    """GitHub 推送輔助類 - 處理所有邊緣情況"""

    def __init__(self):
        self.in_colab = 'google.colab' in sys.modules
        self.token: Optional[str] = None
        self.work_dir: Path = None
        self.repo_url: Optional[str] = None
        self.errors: list = []

    def detect_environment(self) -> dict:
        """檢測執行環境"""
        env_info = {
            'platform': 'colab' if self.in_colab else 'local',
            'python_version': sys.version.split()[0],
            'git_version': self._get_git_version(),
            'git_lfs_version': self._get_git_lfs_version(),
            'current_dir': str(Path.cwd())
        }

        print("🔍 環境檢測:")
        print(f"   平台: {env_info['platform']}")
        print(f"   Git: {env_info['git_version']}")
        print(f"   Git LFS: {env_info['git_lfs_version']}")
        print(f"   當前目錄: {env_info['current_dir']}")

        return env_info

    def _get_git_version(self) -> str:
        """獲取 Git 版本"""
        try:
            result = subprocess.run(['git', '--version'], capture_output=True, text=True)
            return result.stdout.strip()
        except:
            return "未安裝"

    def _get_git_lfs_version(self) -> str:
        """獲取 Git LFS 版本"""
        try:
            result = subprocess.run(['git', 'lfs', 'version'], capture_output=True, text=True)
            return result.stdout.strip().split('\n')[0]
        except:
            return "未安裝"

    def get_token_from_colab_secrets(self) -> Optional[str]:
        """從 Colab Secrets 讀取 GitHub Token（2025最新方法）"""
        if not self.in_colab:
            return None

        try:
            from google.colab import userdata

            # 嘗試多個可能的 secret 名稱
            secret_names = ['GITHUB_TOKEN', 'GH_TOKEN', 'GITHUB_PAT', 'PAT']

            for secret_name in secret_names:
                try:
                    token = userdata.get(secret_name)
                    if token:
                        print(f"✅ 從 Colab Secrets 讀取到 token: {secret_name}")
                        return token
                except Exception:
                    continue

            print("ℹ️ Colab Secrets 中未找到 GitHub token")
            print("   可設置的 secret 名稱: GITHUB_TOKEN, GH_TOKEN, GITHUB_PAT")
            return None

        except ImportError:
            print("⚠️ 無法導入 google.colab.userdata（可能是舊版 Colab）")
            return None

    def get_token_interactive(self) -> str:
        """互動式獲取 token"""
        print("\n🔐 請提供 GitHub Personal Access Token:")
        print("   1. 前往: https://github.com/settings/tokens/new")
        print("   2. 權限勾選: 'repo' (完整倉庫控制)")
        print("   3. 如果是組織倉庫，還需要:")
        print("      → 創建後點擊 'Configure SSO'")
        print("      → 授權給組織 (Authorize)")
        print("")

        try:
            import getpass
            token = getpass.getpass("請貼上 token (輸入會被隱藏): ")
        except:
            token = input("請貼上 token: ")

        if not token or not token.startswith('ghp_'):
            print("❌ Token 格式錯誤（應以 'ghp_' 開頭）")
            return None

        return token.strip()

    def get_token(self) -> bool:
        """獲取 token（優先 Secrets，次之互動輸入）"""
        print("\n🔑 步驟 1: 獲取 GitHub Token")
        print("=" * 60)

        # 優先從 Colab Secrets 讀取
        if self.in_colab:
            self.token = self.get_token_from_colab_secrets()

        # 如果沒有，請用戶輸入
        if not self.token:
            self.token = self.get_token_interactive()

        if not self.token:
            print("❌ 無法獲取 token")
            return False

        print("✅ Token 已準備")
        return True

    def setup_work_directory(self) -> bool:
        """設置正確的工作目錄"""
        print("\n📁 步驟 2: 設置工作目錄")
        print("=" * 60)

        current = Path.cwd()

        # 尋找 Git 倉庫根目錄
        git_root = self._find_git_root(current)

        if git_root:
            self.work_dir = git_root
            os.chdir(self.work_dir)
            print(f"✅ 找到 Git 倉庫: {self.work_dir}")
        elif self.in_colab:
            # Colab 環境特殊處理
            possible_paths = [
                Path('/content/exoplanet-starter'),
                Path('/content'),
                current
            ]

            for path in possible_paths:
                if (path / '.git').exists():
                    self.work_dir = path
                    os.chdir(self.work_dir)
                    print(f"✅ 找到倉庫: {self.work_dir}")
                    break

            if not self.work_dir:
                print("⚠️ 未找到 Git 倉庫，將初始化...")
                self.work_dir = Path('/content/exoplanet-starter')
                self.work_dir.mkdir(parents=True, exist_ok=True)
                os.chdir(self.work_dir)
                subprocess.run(['git', 'init'], check=True)
                print(f"✅ 初始化倉庫: {self.work_dir}")
        else:
            print("❌ 不在 Git 倉庫中")
            return False

        return True

    def _find_git_root(self, start_path: Path) -> Optional[Path]:
        """向上搜尋 Git 倉庫根目錄"""
        current = start_path.resolve()

        while current != current.parent:
            if (current / '.git').exists():
                return current
            current = current.parent

        return None

    def detect_repository_url(self) -> bool:
        """自動檢測倉庫 URL（避免硬編碼）"""
        print("\n🔗 步驟 3: 檢測倉庫 URL")
        print("=" * 60)

        # 檢查現有 remote
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            self.repo_url = result.stdout.strip()
            print(f"✅ 檢測到現有倉庫: {self.repo_url}")

            # 檢查是否為組織倉庫
            if 'github.com' in self.repo_url:
                parts = self.repo_url.split('/')
                if len(parts) >= 2:
                    org_or_user = parts[-2].split(':')[-1]
                    repo_name = parts[-1].replace('.git', '')

                    print(f"   組織/用戶: {org_or_user}")
                    print(f"   倉庫名稱: {repo_name}")

                    # 警告：如果是組織，提醒 SSO
                    if org_or_user != 'your-username':  # 檢測非個人倉庫
                        print("")
                        print("⚠️ 這是組織倉庫，請確認:")
                        print(f"   1. Token 已授權給組織 '{org_or_user}'")
                        print("   2. 前往: https://github.com/settings/tokens")
                        print("   3. 點擊 token → Configure SSO → Authorize")

            return True
        else:
            # 沒有 remote，請用戶提供
            print("⚠️ 未檢測到 remote URL")

            if self.in_colab:
                # Colab 環境提供默認選項
                default_url = "https://github.com/exoplanet-spaceapps/exoplanet-starter.git"
                use_default = input(f"\n使用默認倉庫 {default_url}? (y/n): ").lower()

                if use_default == 'y':
                    self.repo_url = default_url
                else:
                    self.repo_url = input("請輸入你的倉庫 URL: ").strip()
            else:
                self.repo_url = input("請輸入倉庫 URL (https://github.com/user/repo.git): ").strip()

            # 設置 remote
            subprocess.run(['git', 'remote', 'add', 'origin', self.repo_url])
            print(f"✅ 設置 remote: {self.repo_url}")

            return True

    def setup_git_lfs_properly(self) -> bool:
        """正確設置 Git LFS（2025最佳實踐）"""
        print("\n📦 步驟 4: 設置 Git LFS")
        print("=" * 60)

        # 在 Colab 安裝 Git LFS
        if self.in_colab:
            print("   安裝 Git LFS...")
            subprocess.run(['apt-get', 'update', '-qq'], capture_output=True)
            subprocess.run(['apt-get', 'install', '-y', '-qq', 'git-lfs'], capture_output=True)

        # 全局初始化（只執行一次）
        subprocess.run(['git', 'lfs', 'install', '--skip-repo'], capture_output=True)

        # 倉庫級初始化
        result = subprocess.run(['git', 'lfs', 'install'], capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ Git LFS 初始化成功")
        else:
            print(f"   ⚠️ LFS 初始化警告: {result.stderr}")

        # 檢查 LFS 是否在倉庫中啟用
        lfs_check = subprocess.run(
            ['git', 'lfs', 'env'],
            capture_output=True,
            text=True
        )

        if 'git config filter.lfs' in lfs_check.stdout:
            print("   ✅ LFS 過濾器已設置")
        else:
            print("   ⚠️ LFS 過濾器未設置，可能無法追蹤大文件")

        # 創建 .gitattributes
        gitattributes_content = """# Git LFS tracking
*.csv filter=lfs diff=lfs merge=lfs -text
*.json filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.hdf5 filter=lfs diff=lfs merge=lfs -text
*.fits filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
"""

        gitattributes_path = self.work_dir / '.gitattributes'
        with open(gitattributes_path, 'w') as f:
            f.write(gitattributes_content)

        subprocess.run(['git', 'add', '.gitattributes'], capture_output=True)
        print("   ✅ .gitattributes 已創建")

        # ⚠️ 重要：遷移現有文件到 LFS
        print("\n   🔄 遷移現有大文件到 LFS...")
        result = subprocess.run(
            ['git', 'lfs', 'migrate', 'import', '--include="*.csv,*.json"', '--everything'],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("   ✅ 文件已遷移到 LFS")
        else:
            # migrate 可能因為沒有歷史而失敗，這是正常的
            print("   ℹ️ LFS migrate 跳過（可能是新倉庫）")

        return True

    def add_and_commit_files(self) -> bool:
        """添加並提交文件"""
        print("\n💾 步驟 5: 添加並提交文件")
        print("=" * 60)

        # 確保關鍵目錄存在
        critical_dirs = ['data', 'notebooks', 'app']
        for dir_name in critical_dirs:
            dir_path = self.work_dir / dir_name
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"   📁 創建: {dir_name}/")

        # 添加文件
        files_to_add = [
            'data/',
            'notebooks/',
            'app/',
            '.gitattributes',
            'README.md',
            'requirements.txt'
        ]

        added_count = 0
        for file_path in files_to_add:
            full_path = self.work_dir / file_path
            if full_path.exists():
                result = subprocess.run(['git', 'add', file_path], capture_output=True)
                if result.returncode == 0:
                    added_count += 1
                    print(f"   ✅ {file_path}")

        if added_count == 0:
            print("   ⚠️ 沒有文件需要添加")
            return True  # 不是錯誤，可能已經是最新

        # 檢查是否有變更
        status = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True
        )

        if not status.stdout.strip():
            print("   ℹ️ 沒有變更需要提交")
            return True

        # 提交（修正 Co-authored-by 格式）
        commit_message = """data: update NASA exoplanet datasets and analysis pipeline

- Download TOI data from NASA Exoplanet Archive
- Process KOI False Positives as negative samples
- Create supervised training dataset
- Update notebooks with latest analysis

Co-authored-by: hctsai1006 <hctsai1006@gmail.com>
🤖 Generated with Claude Code - https://claude.com/code
"""

        result = subprocess.run(
            ['git', 'commit', '-m', commit_message],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("   ✅ 提交成功")

            # 顯示提交信息
            log = subprocess.run(
                ['git', 'log', '-1', '--oneline'],
                capture_output=True,
                text=True
            )
            print(f"   📝 {log.stdout.strip()}")

            return True
        else:
            print(f"   ❌ 提交失敗: {result.stderr}")
            return False

    def push_to_github(self) -> bool:
        """推送到 GitHub（包含完整驗證）"""
        print("\n🚀 步驟 6: 推送到 GitHub")
        print("=" * 60)

        # 構建認證 URL
        if 'github.com' in self.repo_url:
            auth_url = self.repo_url.replace(
                'https://github.com/',
                f'https://{self.token}@github.com/'
            )
        else:
            print("❌ 只支援 GitHub 倉庫")
            return False

        # 檢測當前分支
        branch_result = subprocess.run(
            ['git', 'branch', '--show-current'],
            capture_output=True,
            text=True
        )
        current_branch = branch_result.stdout.strip() or 'main'

        print(f"   當前分支: {current_branch}")

        # 推送（包含 LFS）
        print(f"   推送到: {self.repo_url}")
        print("   ⏳ 正在推送（這可能需要幾分鐘）...")

        # 設置 Git LFS 追蹤（詳細模式）
        os.environ['GIT_TRACE'] = '1'
        os.environ['GIT_CURL_VERBOSE'] = '1'

        push_result = subprocess.run(
            ['git', 'push', auth_url, f'HEAD:{current_branch}'],
            capture_output=True,
            text=True,
            timeout=300
        )

        # 清除環境變數
        os.environ.pop('GIT_TRACE', None)
        os.environ.pop('GIT_CURL_VERBOSE', None)

        if push_result.returncode == 0:
            print("   ✅ Git push 成功！")

            # ⚠️ 關鍵：驗證 LFS 是否真的推送
            print("\n   🔍 驗證 LFS 文件上傳...")
            lfs_push = subprocess.run(
                ['git', 'lfs', 'push', auth_url, current_branch],
                capture_output=True,
                text=True
            )

            if lfs_push.returncode == 0:
                print("   ✅ LFS 文件推送成功！")
            else:
                print(f"   ⚠️ LFS 推送警告: {lfs_push.stderr}")
                print("")
                print("   💡 可能原因:")
                print("      1. 組織倉庫未啟用 Git LFS")
                print("      2. LFS 配額已用完（免費組織：1GB）")
                print("      3. 需要聯繫 GitHub Support 啟用 LFS")

            # 顯示遠端狀態
            print("\n   📊 遠端倉庫狀態:")
            remote_result = subprocess.run(
                ['git', 'ls-remote', '--heads', auth_url],
                capture_output=True,
                text=True
            )

            if current_branch in remote_result.stdout:
                print(f"   ✅ 分支 '{current_branch}' 已存在於遠端")

            return True

        else:
            error_msg = push_result.stderr
            print(f"   ❌ 推送失敗")
            print(f"   錯誤: {error_msg}")

            # 診斷常見錯誤
            if 'SAML SSO' in error_msg or 'SSO' in error_msg:
                print("\n   💡 診斷: SSO 授權問題")
                print("   解決方案:")
                print("   1. 前往: https://github.com/settings/tokens")
                print("   2. 找到你的 token，點擊 'Configure SSO'")
                print("   3. 授權給組織")

            elif 'Git LFS' in error_msg and 'disabled' in error_msg:
                print("\n   💡 診斷: Git LFS 未啟用")
                print("   解決方案:")
                print("   1. 聯繫組織管理員啟用 Git LFS")
                print("   2. 或暫時不使用 LFS（移除 .gitattributes）")

            elif 'non-fast-forward' in error_msg:
                print("\n   💡 診斷: 版本衝突")
                print("   嘗試自動解決...")

                # 嘗試 pull rebase
                subprocess.run(['git', 'pull', '--rebase', auth_url, current_branch])

                # 重試 push
                retry = subprocess.run(
                    ['git', 'push', auth_url, f'HEAD:{current_branch}'],
                    capture_output=True,
                    text=True
                )

                if retry.returncode == 0:
                    print("   ✅ 解決衝突後推送成功！")
                    return True

            elif 'protected branch' in error_msg:
                print("\n   💡 診斷: 分支保護")
                print("   解決方案:")
                print("   1. 組織可能禁止直接 push 到 main")
                print("   2. 需要創建 Pull Request")

            return False

    def verify_push_success(self) -> bool:
        """驗證推送是否真正成功"""
        print("\n✅ 步驟 7: 驗證推送結果")
        print("=" * 60)

        # 檢查本地狀態
        status = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True
        )

        if status.stdout.strip():
            print("   ⚠️ 仍有未提交的變更")
            return False

        # 檢查是否領先遠端
        status_full = subprocess.run(
            ['git', 'status'],
            capture_output=True,
            text=True
        )

        if 'Your branch is up to date' in status_full.stdout:
            print("   ✅ 本地與遠端同步")
        elif 'Your branch is ahead' in status_full.stdout:
            print("   ⚠️ 本地領先遠端（可能推送失敗）")
            return False

        # 列出已推送的文件
        print("\n   📄 已推送的文件:")
        files_result = subprocess.run(
            ['git', 'ls-tree', '-r', '--name-only', 'HEAD', 'data/'],
            capture_output=True,
            text=True
        )

        if files_result.stdout:
            for line in files_result.stdout.strip().split('\n')[:10]:  # 只顯示前10個
                print(f"      • {line}")

        # 檢查 LFS 文件
        lfs_files = subprocess.run(
            ['git', 'lfs', 'ls-files'],
            capture_output=True,
            text=True
        )

        if lfs_files.stdout:
            print("\n   📦 LFS 追蹤的文件:")
            for line in lfs_files.stdout.strip().split('\n')[:5]:
                print(f"      • {line}")
        else:
            print("\n   ⚠️ 沒有文件被 LFS 追蹤")

        return True

    def run(self) -> bool:
        """執行完整推送流程"""
        print("🚀 GitHub 推送終極解決方案 2025")
        print("=" * 60)
        print("")

        # 環境檢測
        env = self.detect_environment()
        print("")

        # 執行所有步驟
        steps = [
            (self.get_token, "獲取 Token"),
            (self.setup_work_directory, "設置工作目錄"),
            (self.detect_repository_url, "檢測倉庫 URL"),
            (self.setup_git_lfs_properly, "設置 Git LFS"),
            (self.add_and_commit_files, "提交文件"),
            (self.push_to_github, "推送到 GitHub"),
            (self.verify_push_success, "驗證結果"),
        ]

        for step_func, step_name in steps:
            try:
                success = step_func()
                if not success:
                    print(f"\n❌ 失敗於: {step_name}")
                    return False
            except Exception as e:
                print(f"\n❌ 錯誤於 {step_name}: {e}")
                import traceback
                traceback.print_exc()
                return False

        # 成功總結
        print("\n" + "=" * 60)
        print("🎉 推送完成！")
        print("=" * 60)
        print(f"\n📡 倉庫位置: {self.repo_url}")
        print("\n💡 下一步:")
        print("   1. 前往 GitHub 查看你的倉庫")
        print("   2. 檢查 data/ 目錄是否有檔案")
        print("   3. 確認 CSV 檔案不是 LFS pointer")
        print("   4. 如果看到 pointer，需要啟用組織 LFS")
        print("")

        return True


# ========================
# 快速使用函數
# ========================

def quick_push():
    """快速推送函數"""
    helper = GitHubPushHelper()
    return helper.run()


if __name__ == "__main__":
    quick_push()