import argparse
import sys
from pathlib import Path
import datetime

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler


def load_csv(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist")
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"{file_path} is empty")
    return df


def split_train_features_label(df: pd.DataFrame):
    # 学習データ: A列=ID, 最終列=ラベル
    if df.shape[1] < 3:
        raise ValueError("Train CSV must have ID, 1+ feature columns, and label column")
    ids = df.iloc[:, 0].values
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values
    return ids, X, y


def split_test_features(df: pd.DataFrame):
    # テストデータ: A列=ID, 最終列がラベル（存在すれば）
    if df.shape[1] < 2:
        raise ValueError("Test CSV must have ID and at least one feature column")
    ids = df.iloc[:, 0].values
    last_col = str(df.columns[-1]).lower()
    if last_col in ["cluster", "label", "target"]:
        X = df.iloc[:, 1:-1].values
        y = df.iloc[:, -1].values
    else:
        X = df.iloc[:, 1:].values
        y = None
    return ids, X, y


def ensure_valid(y):
    unique = np.unique(y)
    if unique.size < 2:
        raise ValueError("Label column must contain at least two groups")
    return unique


def write_report(report_path: Path, train_accuracy: float, lda: LinearDiscriminantAnalysis, scaler: StandardScaler):
    rows = []

    # 学習判別率
    rows.append(["学習判別率", train_accuracy])

    # 標準化情報
    rows.append(["標準化", "平均=0, 分散=1 (StandardScaler適用済み)"])

    # クラスラベル
    rows.append(["クラスラベル"] + list(lda.classes_))

    # クラス事前確率
    rows.append(["クラス事前確率"] + list(lda.priors_))

    # 各クラス判別関数係数 (ヘッダー付き)
    coef_header = ["クラス"] + [f"coef_{i+1}" for i in range(lda.coef_.shape[1])]
    rows.append(coef_header)
    for i, cl in enumerate(lda.classes_):
        row = [f"クラス {cl}"] + list(lda.coef_[i])
        rows.append(row)

    # 各クラス判別関数閾値
    intercept_header = ["クラス", "intercept"]
    rows.append(intercept_header)
    for i, cl in enumerate(lda.classes_):
        rows.append([f"クラス {cl}", lda.intercept_[i]])

    # 標準化パラメータ
    mean_header = ["標準化平均"] + [f"mean_{i+1}" for i in range(len(scaler.mean_))]
    rows.append(mean_header)
    rows.append(["値"] + list(scaler.mean_))

    var_header = ["標準化分散"] + [f"var_{i+1}" for i in range(len(scaler.var_))]
    rows.append(var_header)
    rows.append(["値"] + list(scaler.var_))

    # CSVとして出力
    df = pd.DataFrame(rows)
    df.to_csv(report_path, index=False, header=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="判別分析スクリプト (LDA)。")
    parser.add_argument("--train", required=True, type=Path, help="学習用CSVファイルパス (ラベルは最終列)")
    parser.add_argument("--test", required=True, type=Path, help="テスト用CSVファイルパス (ラベルは最終列)" )
    parser.add_argument("--output-report", type=Path, default=Path("lda_report.csv"), help="学習レポート出力ファイル (CSV形式)")
    parser.add_argument("--output-test", type=Path, default=Path("test_with_prediction.csv"), help="テスト予測結果出力ファイル")

    args = parser.parse_args()

    # デフォルトファイル名の場合、タイムスタンプを付加して上書きを防ぐ
    if str(args.output_report) == "lda_report.csv":
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        args.output_report = Path(f"lda_report_{timestamp}.csv")
    if str(args.output_test) == "test_with_prediction.csv":
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        args.output_test = Path(f"test_with_prediction_{timestamp}.csv")

    try:
        train_df = load_csv(args.train)
        test_df = load_csv(args.test)

        train_ids, X_train, y_train = split_train_features_label(train_df)
        test_ids, X_test, test_y = split_test_features(test_df)

        ensure_valid(y_train)

        if X_train.shape[1] != X_test.shape[1]:
            raise ValueError(f"Train features ({X_train.shape[1]}) and test features ({X_test.shape[1]}) column count mismatch")

        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)

        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train_std, y_train)

        train_accuracy = lda.score(X_train_std, y_train)

        write_report(args.output_report, train_accuracy, lda, scaler)

        y_pred = lda.predict(X_test_std)

        test_out_df = test_df.copy()
        # ID列を残し、右端に予測ラベルを追加
        test_out_df["predicted_label"] = y_pred
        test_out_df.to_csv(args.output_test, index=False, encoding="utf-8-sig")

        print("===== 判別分析完了 =====")
        print(f"学習データ判別率: {train_accuracy:.6f}")
        print(f"レポート出力: {args.output_report}")
        print(f"テスト出力: {args.output_test}")

    except FileNotFoundError as e:
        print(f"ファイルが見つかりません: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"入力データ検証エラー: {e}", file=sys.stderr)
        sys.exit(1)
    except NotFittedError as e:
        print(f"モデルが未学習です: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"予期せぬエラー: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
