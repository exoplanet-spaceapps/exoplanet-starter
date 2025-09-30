#!/bin/bash
# Notebooks 資料夾清理腳本
# 執行前會先確認並備份檔案

set -e  # 遇到錯誤立即停止

NOTEBOOKS_DIR="C:/Users/tingy/Desktop/dev/exoplanet-starter/notebooks"
ARCHIVE_DIR="$NOTEBOOKS_DIR/archive"

echo "=========================================="
echo "Notebooks 資料夾清理腳本"
echo "=========================================="
echo ""

# 切換到 notebooks 目錄
cd "$NOTEBOOKS_DIR"
echo "📂 當前目錄: $(pwd)"
echo ""

# 創建 archive 目錄結構
echo "📁 創建 archive 目錄..."
mkdir -p archive/outdated_notebooks
mkdir -p archive/old_docs
mkdir -p archive/test_files
echo "✅ Archive 目錄創建完成"
echo ""

# 統計當前檔案數
CURRENT_FILES=$(ls -1 *.ipynb 2>/dev/null | wc -l)
echo "📊 當前 Notebook 檔案數: $CURRENT_FILES"
echo ""

# 移動過時的 Notebook 檔案
echo "🗂️  移動過時的 Notebook 檔案..."
MOVED_NOTEBOOKS=0

for file in \
    "02_bls_baseline_batch.ipynb" \
    "02_bls_baseline_COLAB.ipynb" \
    "02_bls_baseline_COLAB_ENHANCED.ipynb" \
    "02_bls_baseline_LOCAL.ipynb" \
    "03_injection_train.ipynb" \
    "03_injection_train_executed.ipynb" \
    "03_injection_train_MINIMAL_executed_BALANCED.ipynb" \
    "04_newdata_inference_executed.ipynb" \
    "05_metrics_dashboard_executed.ipynb"
do
    if [ -f "$file" ]; then
        mv "$file" archive/outdated_notebooks/
        echo "  ✓ 已移動: $file"
        ((MOVED_NOTEBOOKS++))
    else
        echo "  ⊘ 檔案不存在: $file"
    fi
done
echo "✅ 已移動 $MOVED_NOTEBOOKS 個過時 Notebook 檔案"
echo ""

# 移動文件檔案
echo "📄 移動文件檔案..."
MOVED_DOCS=0

for file in \
    "DIAGNOSIS_REPORT.md" \
    "FIX_NOTEBOOK_03_IMPORTS.md" \
    "STATUS.md" \
    "TEST_RESULTS.md"
do
    if [ -f "$file" ]; then
        mv "$file" archive/old_docs/
        echo "  ✓ 已移動: $file"
        ((MOVED_DOCS++))
    else
        echo "  ⊘ 檔案不存在: $file"
    fi
done
echo "✅ 已移動 $MOVED_DOCS 個文件檔案"
echo ""

# 移動測試檔案
echo "🧪 移動測試檔案..."
MOVED_TESTS=0

for file in \
    "02_bls_baseline_COLAB_PARALLEL.py" \
    "parallel_extraction_module.py" \
    "QUICK_FIX_CELL.py" \
    "quick_test.py" \
    "test_02_simple.py"
do
    if [ -f "$file" ]; then
        mv "$file" archive/test_files/
        echo "  ✓ 已移動: $file"
        ((MOVED_TESTS++))
    else
        echo "  ⊘ 檔案不存在: $file"
    fi
done
echo "✅ 已移動 $MOVED_TESTS 個測試檔案"
echo ""

# 移動 GitHub 工具到 scripts 目錄
echo "🔧 移動 GitHub 工具到 scripts/..."
SCRIPTS_DIR="../scripts"
MOVED_TOOLS=0

for file in \
    "github_push_cell_2025.py" \
    "improved_github_push.py"
do
    if [ -f "$file" ]; then
        if [ -d "$SCRIPTS_DIR" ]; then
            mv "$file" "$SCRIPTS_DIR/"
            echo "  ✓ 已移動: $file -> scripts/"
            ((MOVED_TOOLS++))
        else
            echo "  ⚠ scripts/ 目錄不存在，移動到 archive/test_files/"
            mv "$file" archive/test_files/
        fi
    else
        echo "  ⊘ 檔案不存在: $file"
    fi
done
echo "✅ 已移動 $MOVED_TOOLS 個工具檔案"
echo ""

# 統計結果
echo "=========================================="
echo "📊 清理結果統計"
echo "=========================================="
REMAINING_FILES=$(ls -1 *.ipynb 2>/dev/null | wc -l)
TOTAL_MOVED=$((MOVED_NOTEBOOKS + MOVED_DOCS + MOVED_TESTS + MOVED_TOOLS))

echo "原始檔案數: $CURRENT_FILES 個 Notebooks"
echo "保留檔案數: $REMAINING_FILES 個 Notebooks"
echo "移動總數: $TOTAL_MOVED 個檔案"
echo "  - Notebooks: $MOVED_NOTEBOOKS"
echo "  - 文件: $MOVED_DOCS"
echo "  - 測試: $MOVED_TESTS"
echo "  - 工具: $MOVED_TOOLS"
echo ""

# 顯示保留的檔案
echo "=========================================="
echo "📋 保留的 Notebook 檔案"
echo "=========================================="
ls -lh *.ipynb 2>/dev/null | awk '{print $9, "("$5")"}'
echo ""

# 顯示 archive 大小
echo "=========================================="
echo "💾 Archive 目錄大小"
echo "=========================================="
du -sh archive/
du -sh archive/*/
echo ""

# 顯示保留的輔助檔案
echo "=========================================="
echo "📝 保留的輔助檔案"
echo "=========================================="
ls -lh *.py *.md 2>/dev/null | awk '{print $9, "("$5")"}' || echo "無輔助檔案"
echo ""

echo "=========================================="
echo "✅ 清理完成！"
echo "=========================================="
echo ""
echo "💡 提示："
echo "   - 備份檔案位於: notebooks/archive/"
echo "   - 如需恢復檔案，可從 archive/ 目錄複製回來"
echo "   - 推薦使用: 03_injection_train_FIXED.ipynb"
echo ""