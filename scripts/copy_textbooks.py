#!/usr/bin/env python3
"""
教材拷贝脚本
- 将三本核心教材拷贝到源码目录
- 去除所有图片链接
- 不拷贝图片文件
"""

import re
import shutil
from pathlib import Path


# 源目录
SOURCE_ROOT = Path("/home/firedisposal/nflux")

# 目标目录
PROJECT_ROOT = Path(__file__).parent.parent
TARGET_ROOT = PROJECT_ROOT / "data" / "textbooks"

# 三本核心教材
TEXTBOOKS = [
    "内科护理学",
    "外科护理学", 
    "新编护理学基础",
]

# 图片链接正则（Markdown 格式）
IMAGE_PATTERN = re.compile(r'!\[([^\]]*)\]\([^)]+\)')

# HTML img 标签正则
HTML_IMG_PATTERN = re.compile(r'<img[^>]*>', re.IGNORECASE)


def remove_image_links(content: str) -> str:
    """移除所有图片链接"""
    # 移除 Markdown 图片链接
    content = IMAGE_PATTERN.sub('', content)
    # 移除 HTML img 标签
    content = HTML_IMG_PATTERN.sub('', content)
    # 清理多余空行（连续超过2个空行压缩为2个）
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content.strip()


def copy_textbook(textbook_name: str) -> dict:
    """
    拷贝单本教材
    
    Returns:
        统计信息字典
    """
    source_dir = SOURCE_ROOT / textbook_name
    target_dir = TARGET_ROOT / textbook_name
    
    stats = {
        "name": textbook_name,
        "files": 0,
        "images_removed": 0,
        "skipped": [],
    }
    
    if not source_dir.exists():
        print(f"  ⚠️ 源目录不存在：{source_dir}")
        return stats
    
    # 创建目标目录
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 遍历所有 MD 文件
    for md_file in sorted(source_dir.glob("*.md")):
        try:
            # 读取内容
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 统计图片数量
            img_count = len(IMAGE_PATTERN.findall(content)) + len(HTML_IMG_PATTERN.findall(content))
            
            # 移除图片链接
            cleaned_content = remove_image_links(content)
            
            # 写入目标文件
            target_file = target_dir / md_file.name
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(cleaned_content)
            
            stats["files"] += 1
            stats["images_removed"] += img_count
            
            print(f"  ✓ {md_file.name} (移除 {img_count} 个图片链接)")
            
        except Exception as e:
            print(f"  ❌ 处理失败 {md_file.name}: {e}")
            stats["skipped"].append(md_file.name)
    
    return stats


def main():
    print("=" * 60)
    print("教材拷贝脚本")
    print("=" * 60)
    print(f"源目录：{SOURCE_ROOT}")
    print(f"目标目录：{TARGET_ROOT}")
    print()
    
    # 清理目标目录
    if TARGET_ROOT.exists():
        print("🗑️ 清理旧目录...")
        shutil.rmtree(TARGET_ROOT)
    
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    
    # 拷贝每本教材
    all_stats = []
    for textbook in TEXTBOOKS:
        print(f"\n📚 处理：{textbook}")
        print("-" * 40)
        stats = copy_textbook(textbook)
        all_stats.append(stats)
    
    # 打印统计
    print("\n" + "=" * 60)
    print("📊 拷贝统计")
    print("=" * 60)
    
    total_files = 0
    total_images = 0
    
    for stats in all_stats:
        print(f"\n{stats['name']}:")
        print(f"  文件数：{stats['files']}")
        print(f"  移除图片链接：{stats['images_removed']}")
        if stats['skipped']:
            print(f"  跳过文件：{', '.join(stats['skipped'])}")
        
        total_files += stats['files']
        total_images += stats['images_removed']
    
    print(f"\n总计：")
    print(f"  文件数：{total_files}")
    print(f"  移除图片链接：{total_images}")
    print(f"\n✅ 教材已拷贝至：{TARGET_ROOT}")


if __name__ == "__main__":
    main()