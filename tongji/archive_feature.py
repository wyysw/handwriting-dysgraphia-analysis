import os
import json
import pandas as pd

LABEL_PATH = "data/labels.csv"
GAMES = ["sym", "circle", "maze"]

FEATURE_COLS = ["F1", "F2", "F3", "F4", "C1", "C2", "C3"]

def load_json_features(game, sample_id):
    json_path = f"output_{game}/extract/{sample_id}.json"
    if not os.path.exists(json_path):
        print(f"[Warning] Missing file: {json_path}")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取特征
    features = {k: data.get(k, None) for k in FEATURE_COLS}
    return features


def process_game(df_game, game):
    rows = []

    for _, row in df_game.iterrows():
        sample_id = row["sample_id"]
        label = row["label"]

        features = load_json_features(game, sample_id)
        if features is None:
            continue

        record = {
            "sample_id": sample_id,
            "game": game,
            "label": label,
            **features
        }
        rows.append(record)

    df_out = pd.DataFrame(rows)
    return df_out


def main():
    labels = pd.read_csv(LABEL_PATH)

    all_dfs = []

    for game in GAMES:
        df_game = labels[labels["game"] == game]

        df_result = process_game(df_game, game)

        # 保存每个 game 的 csv
        out_path = f"data/feature/{game}.csv"
        df_result.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[Saved] {out_path}")

        all_dfs.append(df_result)

    # 合并所有 game
    df_all = pd.concat(all_dfs, ignore_index=True)
    all_path = "data/feature/all.csv"
    df_all.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] {all_path}")


if __name__ == "__main__":
    main()
