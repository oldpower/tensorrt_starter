import pandas as pd
import pickle
from pathlib import Path
from urllib.parse import unquote
from sqlalchemy import create_engine,text
import os
from collections import Counter

def create_sql_engine():
    engine = create_engine(
        f"mysql+pymysql://root@192.168.103.204:19030/reference?charset=utf8",
        pool_size=20, max_overflow=20, pool_pre_ping=True, pool_recycle=True)
    return engine

def get_journal_from_doi(engine,DOI:str):
    sql = text("SELECT Journal FROM scimag WHERE DOI = :doi")
    df = pd.read_sql(sql, engine, params={"doi": DOI})
    return df

def load_extra_status(pth):
    """加载提取状态"""
    if os.path.exists(pth):
        with open(pth, "rb") as f:
            return pickle.load(f)
    return {}

def save_extra_status(status,pth):
    """保存提取状态"""
    with open(pth, "wb") as f:
        pickle.dump(status, f)

def count_journals():
    
    output_base_dir = './assets/step4_dirtxt'
    engine = create_sql_engine()
    pkl_dir = './assets/count_journals.pkl'

    log_dir = Path(output_base_dir)
    if not log_dir.exists():
        print(f"⚠️ 目录 {output_base_dir} 不存在")
        return

    # 查找所有 *_list.txt 文件
    txt_files = list(log_dir.glob("*_list.txt"))
    if not txt_files:
        print(f"⚠️ 在 {output_base_dir} 中未找到任何 *_list.txt 文件")
        return

    count_dict = load_extra_status(pkl_dir)
    for txt_file in txt_files:
        # 从文件名提取子目录名，例如 "10.1016_list.txt" → "10.1016"
        subdir_name = txt_file.stem.replace("_list", "")
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
                non_empty_lines = [line for line in lines if line.strip()]

            #
            for pdf in non_empty_lines:
                path = pdf.strip()
                if path:
                    decodepdfname = unquote(path)
                    p_path = Path(decodepdfname)
                    doi = f"{p_path.parent.name}/{p_path.stem}"
                    if doi in count_dict:
                        print(f"SKIP {doi}")
                        continue
                    df_journal = get_journal_from_doi(engine,doi)
                    if len(df_journal):
                        Journal = df_journal['Journal'][0]
                        count_dict[doi] = Journal
                        print(f"{doi:<50} \t {Journal}")
                    else:
                        count_dict[doi] = ""
                        print(f"{doi} \t ")
        except KeyboardInterrupt:
            print("\n⚠️ 收到 Ctrl+C，正在保存进度...")
            save_extra_status(count_dict, pkl_dir)
            print("✅ 状态已保存，程序退出。")
            exit(0)  # 或 sys.exit(0)
        except Exception as e:
            print(f"❌ 读取 {txt_file} 出错: {e}")

        save_extra_status(count_dict,pkl_dir)

def plot_top_counts(counts, top_n=50):
    import matplotlib.pyplot as plt
    # 获取前top_n项
    top_items = dict(list(counts.items())[:top_n])

    # 准备数据
    names = list(top_items.keys())
    counts = list(top_items.values())

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(10, 8)) # 根据需要调整图形大小

    # 绘制柱状图
    bars = ax.barh(names, counts, color='skyblue')
    ax.invert_yaxis()  # 按降序显示

    # 添加计数值作为标签
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{int(width)}',
                    xy=(width, bar.get_y() + bar.get_height() / 2),
                    xytext=(3, 0),  # 3 points horizontal offset
                    textcoords="offset points",
                    ha='left', va='center')

    # 设置标题和标签
    ax.set_title(f'Top {top_n} Subdirectories by PDF Count')
    ax.set_xlabel('PDF Counts')
    ax.set_ylabel('Subdirectory Names')

    # 显示图表
    plt.tight_layout()
    plt.show()

def journals_num():
    # # pd方法
    # df = pd.DataFrame(list(data.items()), columns=['DOI', 'journal'])
    # df['journal'] = df['journal'].str.strip().str.title()  # 或 .str.lower()
    # journal_counts = df['journal'].value_counts()


    # # 字典方法
    # from collections import Counter
    # data_cleaned  = {doi: journal.strip().title() for doi, journal in data_dict.items()}
    # counts = Counter(data_cleaned .values())
    # counts_dict = dict(counts)
    # sorted_counts = dict(sorted(count_dict.items(), key=lambda x: x[1], reverse=True))

    #可选 转为按频次降序的字典

    pkl_dir = './assets/count_journals.pkl'
    data_dict = load_extra_status(pkl_dir)
    data_cleaned  = {doi: journal.strip().title() for doi, journal in data_dict.items()}
    print(f"找到 {len(data_cleaned)} 个 DOI:")
    target_journal = "Bioorganic & Medicinal Chemistry"
    dois = [doi for doi, journal in data_cleaned.items() if journal == target_journal]
    for doi in dois:
        print(doi)
    exit()
    counts = Counter(data_cleaned .values())
    sorted_counts = dict(counts.most_common())  # 推荐：简洁且高效

    plot_top_counts(sorted_counts, top_n=51)

    top5 = list(sorted_counts.items())[:5]
    print("Top 5:", top5)
    print(f"Journal number:{len(sorted_counts)}")

    df = pd.DataFrame(list(sorted_counts.items()), columns=['journal', 'num'])
    df.to_csv('./assets/journals_num.csv', index=False)


if __name__ == "__main__":
    # count_journals()
    journals_num()




