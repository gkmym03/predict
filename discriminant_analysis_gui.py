import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sys
import os
from pathlib import Path
import datetime
import subprocess

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler


def load_csv(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist")
    df = pd.read_csv(file_path, encoding='utf-8')
    if df.empty:
        raise ValueError(f"{file_path} is empty")
    return df


def split_train_features_label(df: pd.DataFrame):
    if df.shape[1] < 3:
        raise ValueError("Train CSV must have ID, 1+ feature columns, and label column")
    ids = df.iloc[:, 0].values
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values
    return ids, X, y


def split_test_features(df: pd.DataFrame):
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


def run_analysis(train_path, test_path, report_path, test_out_path):
    try:
        train_df = load_csv(Path(train_path))
        test_df = load_csv(Path(test_path))

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

        write_report(Path(report_path), train_accuracy, lda, scaler)

        y_pred = lda.predict(X_test_std)

        test_out_df = test_df.copy()
        test_out_df["predicted_label"] = y_pred
        test_out_df.to_csv(Path(test_out_path), index=False, encoding="utf-8-sig")

        return f"成功: 学習判別率 {train_accuracy:.6f}\nレポート: {report_path}\nテスト出力: {test_out_path}"

    except Exception as e:
        return f"エラー: {str(e)}"


class DiscriminantAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("判別分析ツール")
        self.root.geometry("600x400")

        # 学習データ選択
        ttk.Label(root, text="学習データCSV:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.train_entry = ttk.Entry(root, width=50)
        self.train_entry.grid(row=0, column=1, padx=10, pady=5)
        ttk.Button(root, text="選択", command=self.select_train).grid(row=0, column=2, padx=10, pady=5)

        # テストデータ選択
        ttk.Label(root, text="テストデータCSV:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.test_entry = ttk.Entry(root, width=50)
        self.test_entry.grid(row=1, column=1, padx=10, pady=5)
        ttk.Button(root, text="選択", command=self.select_test).grid(row=1, column=2, padx=10, pady=5)

        # レポート出力
        ttk.Label(root, text="レポート出力:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
        self.report_entry = ttk.Entry(root, width=50)
        self.report_entry.grid(row=2, column=1, padx=10, pady=5)
        ttk.Button(root, text="選択", command=self.select_report).grid(row=2, column=2, padx=10, pady=5)

        # テスト出力
        ttk.Label(root, text="テスト出力:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        self.test_out_entry = ttk.Entry(root, width=50)
        self.test_out_entry.grid(row=3, column=1, padx=10, pady=5)
        ttk.Button(root, text="選択", command=self.select_test_out).grid(row=3, column=2, padx=10, pady=5)

        # 実行ボタン
        self.run_button = ttk.Button(root, text="実行", command=self.run_analysis)
        self.run_button.grid(row=4, column=1, pady=20)

        # 結果表示
        self.result_text = tk.Text(root, height=10, width=70)
        self.result_text.grid(row=5, column=0, columnspan=3, padx=10, pady=10)

        # デフォルトファイル名設定
        self.set_default_outputs()

    def set_default_outputs(self):
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        self.report_entry.insert(0, f"lda_report_{timestamp}.csv")
        self.test_out_entry.insert(0, f"test_with_prediction_{timestamp}.csv")

    def select_train(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.train_entry.delete(0, tk.END)
            self.train_entry.insert(0, file_path)

    def select_test(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.test_entry.delete(0, tk.END)
            self.test_entry.insert(0, file_path)

    def select_report(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.report_entry.delete(0, tk.END)
            self.report_entry.insert(0, file_path)

    def select_test_out(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.test_out_entry.delete(0, tk.END)
            self.test_out_entry.insert(0, file_path)

    def run_analysis(self):
        train_path = self.train_entry.get()
        test_path = self.test_entry.get()
        report_path = self.report_entry.get()
        test_out_path = self.test_out_entry.get()

        if not train_path or not test_path:
            messagebox.showerror("エラー", "学習データとテストデータを選択してください")
            return

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "処理中...\n")
        self.root.update()

        result = run_analysis(train_path, test_path, report_path, test_out_path)
        self.result_text.insert(tk.END, result)


if __name__ == "__main__":
    root = tk.Tk()
    app = DiscriminantAnalysisGUI(root)
    root.mainloop()
