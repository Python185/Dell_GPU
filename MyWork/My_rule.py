# Excelを読込んで、Webからデータを取得する

import pandas as pd
import os
import requests
import json
from openpyxl import load_workbook
from bs4 import BeautifulSoup
import re
from datetime import datetime
import schedule
import time
import logging

# ログの設定
def setup_logging():
    """ログ設定を行う関数"""
    # 既存のログハンドラーをクリア
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    log_filename = f"price_update_log_{datetime.now().strftime('%Y%m')}.log"
    
    # ログディレクトリを作成（存在しない場合）
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ログを設定
logger = setup_logging()

def read_excel_all_sheets(file_path):
    """
    Excelファイルの全シートを読み込む関数
    
    Parameters:
    file_path (str): Excelファイルのパス
    
    Returns:
    dict: シート名をキー、DataFrameを値とする辞書
    """
    try:
        # Excelファイルの全シートを読み込み
        excel_data = pd.read_excel(file_path, sheet_name=None)
        
        print(f"ファイル '{file_path}' を読み込みました。")
        print(f"シート数: {len(excel_data)}")
        print(f"シート名: {list(excel_data.keys())}")
        
        # 各シートの基本情報を表示
        for sheet_name, df in excel_data.items():
            print(f"\n--- シート '{sheet_name}' ---")
            print(f"形状: {df.shape}")
            print(f"列名: {df.columns.tolist()}")
            print("最初の3行:")
            print(df.head(3))
            print("-" * 50)
        
        return excel_data
    
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

def get_usd_jpy_rate():
    """
    Google FinanceからUSD/JPYの為替レートを取得する関数
    
    Returns:
    float: USD/JPYレート
    """
    try:
        # Google FinanceのUSD/JPYページ
        url = "https://www.google.com/finance/quote/USD-JPY"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        
        # BeautifulSoupでHTMLを解析
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Google Financeの価格表示要素を検索
        # 複数のパターンで価格を探す
        price_patterns = [
            {'class': 'YMlKec fxKbKc'},  # 一般的なGoogle Financeの価格クラス
            {'data-last-price': True},   # data-last-price属性
            {'class': 'kf1m0'},          # 別の価格クラス
            {'class': 'YMlKec'},         # シンプルなクラス
        ]
        
        usd_jpy_rate = None
        
        for pattern in price_patterns:
            price_element = soup.find('div', pattern)
            if price_element:
                try:
                    price_text = price_element.get_text().strip()
                    # 数字のみを抽出（カンマを除去）
                    price_match = re.search(r'(\d{1,3}(?:\.\d{2})?)', price_text.replace(',', ''))
                    if price_match:
                        potential_rate = float(price_match.group(1))
                        if 100 <= potential_rate <= 200:  # 妥当なUSD/JPYレートの範囲
                            usd_jpy_rate = potential_rate
                            break
                except:
                    continue
        
        # パターンマッチが失敗した場合、テキスト全体から抽出を試行
        if not usd_jpy_rate:
            text_content = soup.get_text()
            # USD/JPYレートのパターンを検索
            rate_patterns = [
                r'USD.*?JPY.*?(\d{3}\.\d{2})',
                r'(\d{3}\.\d{2}).*?JPY',
                r'USD.*?(\d{3}\.\d{2})',
            ]
            
            for pattern in rate_patterns:
                rate_match = re.search(pattern, text_content)
                if rate_match:
                    potential_rate = float(rate_match.group(1))
                    if 100 <= potential_rate <= 200:  # 妥当な範囲チェック
                        usd_jpy_rate = potential_rate
                        break
        
        if usd_jpy_rate:
            return usd_jpy_rate
        else:
            print("Google Financeから価格を抽出できませんでした。デフォルトレートを使用します。")
            return 150.0  # デフォルトレート
            
    except Exception as e:
        print(f"Google Finance取得エラー: {e}")
        print("デフォルトレートを使用します。")
        return 150.0  # デフォルトレート

def get_crypto_prices():
    """
    CoinGeckoから仮想通貨の価格を取得する関数
    
    Returns:
    dict: BTCとETHの価格（万円単位）
    """
    try:
        # CoinGecko APIを使用（無料、APIキー不要）
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=jpy"
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # HTTPエラーをチェック
        data = response.json()
        
        # データの妥当性をチェック
        if 'bitcoin' not in data or 'ethereum' not in data:
            raise ValueError("CoinGecko APIから必要なデータが取得できませんでした")
        
        btc_jpy = data['bitcoin']['jpy'] / 10000  # 万円単位
        eth_jpy = data['ethereum']['jpy'] / 10000  # 万円単位
        
        # 価格の妥当性をチェック
        if btc_jpy <= 0 or eth_jpy <= 0:
            raise ValueError("無効な価格データが取得されました")
        
        return {
            'BTC': btc_jpy,
            'ETH': eth_jpy
        }
    
    except Exception as e:
        print(f"CoinGecko取得エラー: {e}")
        print("デフォルト価格を使用します。")
        return {'BTC': 600.0, 'ETH': 40.0}  # デフォルト価格

def get_precious_metal_prices():
    """
    MetalPriceAPIから貴金属価格を取得する関数
    
    Returns:
    dict: 金、銀、プラチナの価格（円/g）
    """
    # MetalPriceAPIから取得
    try:
        return get_metalpriceapi_precious_metal_prices()
    except:
        # デフォルト価格を使用
        return {
            'gold': 22000.0,
            'platinum': 8000.0,
            'silver': 286.0
        }

def get_metalpriceapi_precious_metal_prices():
    """
    MetalPriceAPIから金、銀、プラチナの価格を取得する関数
    
    Returns:
    dict: 金、銀、プラチナの価格（円/g）
    """
    try:
        # 現在のUSD/JPYレートを取得
        usd_jpy_rate = get_usd_jpy_rate()
        
        # 貴金属はオンス（avoirdupois ounce）で表示される
        oz_to_gram = 28.35
        
        # MetalPriceAPIのエンドポイント
        api_key = "0f87f8e218bb8fb3c0b40c46d56911a8"
        api_url = "https://api.metalpriceapi.com/v1/latest"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'x-api-key': api_key
        }
        
        params = {
            'base': 'USD',
            'currencies': 'XAU,XAG,XPT'
        }
        
        response = requests.get(api_url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        # MetalPriceAPIのレスポンスから価格を取得
        prices = {}
        
        # ratesが文字列の場合、JSONとしてパース
        if 'rates' in data:
            if isinstance(data['rates'], str):
                import json
                rates = json.loads(data['rates'])
            else:
                rates = data['rates']
            
            # 金の価格（USD/オンス）
            if 'USDXAU' in rates:
                gold_usd_per_oz = float(rates['USDXAU'])
                prices['gold'] = (gold_usd_per_oz * usd_jpy_rate) / oz_to_gram
            
            # 銀の価格（USD/オンス）
            if 'USDXAG' in rates:
                silver_usd_per_oz = float(rates['USDXAG'])
                prices['silver'] = (silver_usd_per_oz * usd_jpy_rate) / oz_to_gram
            
            # プラチナの価格（USD/オンス）
            if 'USDXPT' in rates:
                platinum_usd_per_oz = float(rates['USDXPT'])
                prices['platinum'] = (platinum_usd_per_oz * usd_jpy_rate) / oz_to_gram
        
        # 全ての価格が取得できなかった場合はエラー
        if len(prices) < 3:
            raise Exception("貴金属価格取得不完全")
        
        return prices
        
    except:
        raise
        


def update_crypto_prices_in_excel(file_path):
    """
    Excelファイルの仮想通貨価格を更新する関数（書式を保持）
    
    Parameters:
    file_path (str): Excelファイルのパス
    """
    try:
        # 現在の仮想通貨価格を取得
        crypto_prices = get_crypto_prices()
        
        # openpyxlでExcelファイルを読み込み（書式を保持）
        workbook = load_workbook(file_path)
        
        # calcシートを取得
        if 'calc' in workbook.sheetnames:
            worksheet = workbook['calc']
            
            # J29（ビットコイン価格）とJ30（イーサリウム価格）を更新
            # openpyxlでは行と列は1ベース
            btc_cell = 'J29'
            eth_cell = 'J30'
            
            # セルの値のみを更新（書式は保持）
            worksheet[btc_cell].value = crypto_prices['BTC']
            worksheet[eth_cell].value = crypto_prices['ETH']
            
            print(f"J29（ビットコイン価格）を {crypto_prices['BTC']:.2f}万円 に更新しました")
            print(f"J30（イーサリウム価格）を {crypto_prices['ETH']:.2f}万円 に更新しました")
            
            # 書式を保持したまま保存
            workbook.save(file_path)
            print(f"ファイル '{file_path}' の更新が完了しました（書式保持）。")
            
        else:
            print("エラー: 'calc' シートが見つかりません")
            
    except Exception as e:
        print(f"Excel更新エラー: {e}")
    finally:
        # ワークブックを閉じる
        if 'workbook' in locals():
            workbook.close()

def update_precious_metal_prices_in_excel(file_path):
    """
    Excelファイルの貴金属価格を更新する関数（書式を保持）
    
    Parameters:
    file_path (str): Excelファイルのパス
    """
    workbook = None
    try:
        print(f"デバッグ: Excelファイルパス = {file_path}")
        
        # ファイルの存在確認
        if not os.path.exists(file_path):
            raise Exception(f"ファイルが存在しません: {file_path}")
        
        # 貴金属価格を取得（既に取得済みの価格を使用）
        print("デバッグ: 貴金属価格を取得中...")
        metal_prices = get_precious_metal_prices()
        print(f"デバッグ: 取得した価格 = {metal_prices}")
        
        # 価格データの妥当性チェック
        if not metal_prices or not all(key in metal_prices for key in ['gold', 'silver', 'platinum']):
            raise Exception("貴金属価格データが不完全です")
        
        # openpyxlでExcelファイルを読み込み（書式を保持）
        print("デバッグ: Excelファイルを読み込み中...")
        workbook = load_workbook(file_path)
        print(f"デバッグ: 利用可能なシート名 = {workbook.sheetnames}")
        
        # ストックシートを取得
        if 'ストック' in workbook.sheetnames:
            worksheet = workbook['ストック']
            print("デバッグ: 'ストック'シートを取得しました")
            
            # 更新前の値を確認
            print("デバッグ: 更新前の値を確認...")
            old_gold = worksheet['R30'].value
            old_silver = worksheet['R31'].value
            old_platinum = worksheet['R32'].value
            print(f"デバッグ: 更新前 - 金:{old_gold}, 銀:{old_silver}, プラチナ:{old_platinum}")
            
            # R31（金価格）、R32（銀価格）、R33（プラチナ価格）を更新
            gold_cell = 'R31'
            silver_cell = 'R32'
            platinum_cell = 'R33'
            
            # セルの値のみを更新（書式は保持）
            worksheet[gold_cell].value = metal_prices['gold']
            worksheet[silver_cell].value = metal_prices['silver']
            worksheet[platinum_cell].value = metal_prices['platinum']
            
            # 更新後の値を確認
            print("デバッグ: 更新後の値を確認...")
            new_gold = worksheet['R30'].value
            new_silver = worksheet['R31'].value
            new_platinum = worksheet['R32'].value
            print(f"デバッグ: 更新後 - 金:{new_gold}, 銀:{new_silver}, プラチナ:{new_platinum}")
            
            print(f"R30（金買取価格）を {metal_prices['gold']:.2f}円/g に更新しました")
            print(f"R31（銀買取価格）を {metal_prices['silver']:.2f}円/g に更新しました")
            print(f"R32（プラチナ買取価格）を {metal_prices['platinum']:.2f}円/g に更新しました")
            
            # 書式を保持したまま保存
            print("デバッグ: ファイルを保存中...")
            workbook.save(file_path)
            print(f"ストックシートの貴金属価格更新が完了しました（書式保持）。")
            
            # 保存後の確認
            print("デバッグ: 保存後の確認...")
            workbook.close()
            workbook = None
            
            # ファイルを再度開いて確認
            verify_workbook = load_workbook(file_path)
            verify_worksheet = verify_workbook['ストック']
            verify_gold = verify_worksheet['R30'].value
            verify_silver = verify_worksheet['R31'].value
            verify_platinum = verify_worksheet['R32'].value
            verify_workbook.close()
            
            print(f"デバッグ: 保存後確認 - 金:{verify_gold}, 銀:{verify_silver}, プラチナ:{verify_platinum}")
            
        else:
            available_sheets = workbook.sheetnames
            raise Exception(f"'ストック' シートが見つかりません。利用可能なシート: {available_sheets}")
            
    except Exception as e:
        print(f"貴金属価格Excel更新エラー: {e}")
        import traceback
        print(f"詳細エラー情報: {traceback.format_exc()}")
        raise  # エラーを再発生させて上位で捕捉できるようにする
    finally:
        # ワークブックを閉じる
        if workbook is not None:
            try:
                workbook.close()
            except:
                pass

def update_execution_date_in_excel(file_path):
    """
    家計資産シートのH14、H16に実行日を更新する関数（書式を保持）
    
    Parameters:
    file_path (str): Excelファイルのパス
    """
    try:
        # 現在の日付を取得
        current_date = datetime.now()
        date_str = current_date.strftime("%Y/%m/%d")
        
        # openpyxlでExcelファイルを読み込み（書式を保持）
        workbook = load_workbook(file_path)
        
        # 家計資産シートを取得
        if '家計資産' in workbook.sheetnames:
            worksheet = workbook['家計資産']
            
            # H14とH16に日付を更新（実際のセル位置）
            h14_cell = 'H14'
            h16_cell = 'H16'
            
            # セルの値のみを更新（書式は保持）
            worksheet[h14_cell].value = date_str
            worksheet[h16_cell].value = date_str
            
            print(f"H14に実行日 {date_str} を更新しました")
            print(f"H16に実行日 {date_str} を更新しました")
            
            # 書式を保持したまま保存
            workbook.save(file_path)
            print(f"家計資産シートの実行日更新が完了しました（書式保持）。")
            
        else:
            print("エラー: '家計資産' シートが見つかりません")
            
    except Exception as e:
        print(f"実行日Excel更新エラー: {e}")
    finally:
        # ワークブックを閉じる
        if 'workbook' in locals():
            workbook.close()

def update_all_prices_with_logging(file_path):
    """
    全ての価格を更新し、ログを記録する関数
    
    Parameters:
    file_path (str): Excelファイルのパス
    """
    logger.info("=== 価格データの自動更新を開始 ===")
    
    try:
        # ファイルが存在するかチェック
        if not os.path.exists(file_path):
            logger.error(f"ファイルが見つかりません: {file_path}")
            return False
        
        success_count = 0
        total_tasks = 3
        
        # 仮想通貨価格を更新
        logger.info("--- 仮想通貨価格の更新 ---")
        try:
            update_crypto_prices_in_excel(file_path)
            success_count += 1
            logger.info("仮想通貨価格の更新が完了しました")
        except Exception as e:
            logger.error(f"仮想通貨価格更新エラー: {e}")
        
        # 貴金属価格を更新
        logger.info("--- 貴金属価格の更新 ---")
        try:
            update_precious_metal_prices_in_excel(file_path)
            success_count += 1
            logger.info("貴金属価格の更新が完了しました")
        except Exception as e:
            logger.error(f"貴金属価格更新エラー: {e}")
            import traceback
            logger.error(f"詳細エラー情報: {traceback.format_exc()}")
        
        # 実行日を更新
        logger.info("--- 実行日の更新 ---")
        try:
            update_execution_date_in_excel(file_path)
            success_count += 1
            logger.info("実行日の更新が完了しました")
        except Exception as e:
            logger.error(f"実行日更新エラー: {e}")
        
        # 結果をログに記録
        logger.info(f"=== 更新完了: {success_count}/{total_tasks} タスクが成功 ===")
        
        if success_count == total_tasks:
            logger.info("全ての更新が正常に完了しました")
            return True
        else:
            logger.warning(f"{total_tasks - success_count} 個のタスクが失敗しました")
            return False
            
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        return False

def scheduled_job():
    """スケジュールされたジョブを実行する関数"""
    file_path = "datasets/maki_250630.xlsx"
    logger.info("定期実行ジョブが開始されました")
    
    try:
        success = update_all_prices_with_logging(file_path)
        if success:
            logger.info("定期実行ジョブが正常に完了しました")
        else:
            logger.warning("定期実行ジョブが部分的に失敗しました")
    except Exception as e:
        logger.error(f"定期実行ジョブでエラーが発生しました: {e}")

def run_scheduler():
    """スケジューラを実行する関数"""
    logger.info("スケジューラを開始します - 毎日朝3:00に実行")
    
    # 毎日朝3時に実行するようにスケジュール
    schedule.every().day.at("03:00").do(scheduled_job)
    
    # 即座にテスト実行（オプション）
    logger.info("初回テスト実行を行います...")
    scheduled_job()
    
    logger.info("次回実行予定: 明日の朝3:00")
    logger.info("スケジューラが開始されました。Ctrl+Cで停止できます。")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # 1分ごとにチェック
    except KeyboardInterrupt:
        logger.info("スケジューラが手動で停止されました")
    except Exception as e:
        logger.error(f"スケジューラでエラーが発生しました: {e}")

def create_batch_file():
    """
    Windowsタスクスケジューラ用のバッチファイルを作成する関数
    """
    batch_content = f'''@echo off
cd /d "{os.getcwd()}"
python My_rule.py --auto
pause
'''
    
    batch_file_path = "run_price_update.bat"
    try:
        with open(batch_file_path, 'w', encoding='shift_jis') as f:
            f.write(batch_content)
        logger.info(f"バッチファイルを作成しました: {batch_file_path}")
        logger.info("このバッチファイルをWindowsタスクスケジューラに登録できます")
        return batch_file_path
    except Exception as e:
        logger.error(f"バッチファイル作成エラー: {e}")
        return None

def show_task_scheduler_instructions():
    """Windowsタスクスケジューラの設定手順を表示する関数"""
    instructions = """
=== Windowsタスクスケジューラの設定手順 ===

1. Windowsキー + R を押して「taskschd.msc」と入力してタスクスケジューラを開く

2. 右側の「基本タスクの作成」をクリック

3. 基本タスクの作成ウィザードで以下を設定：
   - 名前: 「価格データ自動更新」
   - 説明: 「Excel価格データの自動更新（毎日朝3時）」

4. トリガー: 「毎日」を選択

5. 開始日時: 今日の日付、時刻は「3:00:00」に設定

6. 操作: 「プログラムの開始」を選択

7. プログラム/スクリプト: 作成されたバッチファイルのフルパスを指定
   例: C:\\path\\to\\run_price_update.bat

8. 「完了」をクリック

9. 作成されたタスクを右クリック → 「プロパティ」で詳細設定：
   - 「ユーザーがログオンしているかどうかにかかわらず実行する」を選択
   - 「最上位の特権で実行する」にチェック

=== 注意事項 ===
- PCが朝3時に起動していることを確認してください
- スリープモードの場合は「タスクを実行するためにスリープを解除する」にチェック
- ネットワーク接続が必要です
"""
    print(instructions)
    logger.info("Windowsタスクスケジューラの設定手順を表示しました")

# メイン実行部分
if __name__ == "__main__":
    import sys
    
    # ファイルパスを指定
    file_path = "MyWork/datasets/maki_250630.xlsx"
    
    # ファイルが存在しない場合の代替パスを試行
    if not os.path.exists(file_path):
        print(f"指定されたファイルが見つかりません: {file_path}")
        print("代替パスを検索中...")
        
        # MyWorkディレクトリ内を確認
        alternative_path = os.path.join("MyWork", "datasets", "maki_250630.xlsx")
        if os.path.exists(alternative_path):
            file_path = alternative_path
            print(f"代替ファイルを発見: {file_path}")
        else:
            # 現在のディレクトリから相対パスで検索
            found = False
            for root, dirs, files in os.walk("."):
                if "maki_250630.xlsx" in files:
                    file_path = os.path.join(root, "maki_250630.xlsx")
                    print(f"ファイルを発見: {file_path}")
                    found = True
                    break
            
            if not found:
                print("エラー: maki_250630.xlsx ファイルが見つかりません")
                print("利用可能なExcelファイル:")
                for root, dirs, files in os.walk("."):
                    for file in files:
                        if file.endswith('.xlsx'):
                            print(f"  - {os.path.join(root, file)}")
                sys.exit(1)
    
    # コマンドライン引数をチェック
    if len(sys.argv) > 1:
        if sys.argv[1] == "--scheduler":
            # スケジューラモードで実行
            run_scheduler()
        elif sys.argv[1] == "--auto":
            # 自動実行モード（ログ付き）
            update_all_prices_with_logging(file_path)
        elif sys.argv[1] == "--setup":
            # セットアップモード（バッチファイル作成とタスクスケジューラ手順表示）
            logger.info("セットアップモードで実行します")
            batch_file = create_batch_file()
            if batch_file:
                show_task_scheduler_instructions()
        else:
            print("使用方法:")
            print("  python My_rule.py                 # 通常実行")
            print("  python My_rule.py --scheduler     # スケジューラで常駐実行")
            print("  python My_rule.py --auto          # 自動実行（ログ付き）")
            print("  python My_rule.py --setup         # セットアップ（バッチファイル作成）")
    else:
        # 通常実行（既存の動作）
        # ファイルが存在するかチェック
        if os.path.exists(file_path):
            print("=== 価格データの更新を開始 ===")
            
            # 仮想通貨価格を更新
            print("\n--- 仮想通貨価格の更新 ---")
            update_crypto_prices_in_excel(file_path)
            
            # 貴金属価格を更新
            print("\n--- 貴金属価格の更新 ---")
            try:
                update_precious_metal_prices_in_excel(file_path)
            except Exception as e:
                print(f"貴金属価格更新でエラーが発生しました: {e}")
                import traceback
                print(f"詳細エラー情報: {traceback.format_exc()}")
            
            # 実行日を更新
            print("\n--- 実行日の更新 ---")
            update_execution_date_in_excel(file_path)
            
            print("\n=== 更新後のファイル確認 ===")
            # 更新後のデータを確認
            all_sheets = read_excel_all_sheets(file_path)
            
            if all_sheets:
                # calcシートの確認
                if 'calc' in all_sheets:
                    try:
                        workbook_check = load_workbook(file_path)
                        worksheet_check = workbook_check['calc']
                        j29_value = worksheet_check['J29'].value
                        j30_value = worksheet_check['J30'].value
                        workbook_check.close()
                        
                        print(f"\ncalcシート更新後:")
                        print(f"J29（ビットコイン）: {j29_value}")
                        print(f"J30（イーサリウム）: {j30_value}")
                    except Exception as e:
                        print(f"calcシート確認エラー: {e}")
                
                # ストックシートの確認
                if 'ストック' in all_sheets:
                    try:
                        workbook_check = load_workbook(file_path)
                        worksheet_check = workbook_check['ストック']
                        r30_value = worksheet_check['R30'].value
                        r31_value = worksheet_check['R31'].value
                        r32_value = worksheet_check['R32'].value
                        workbook_check.close()
                        
                        print(f"\nストックシート更新後:")
                        print(f"R30（金買取価格）: {r30_value}")
                        print(f"R31（銀買取価格）: {r31_value}")
                        print(f"R32（プラチナ買取価格）: {r32_value}")
                    except Exception as e:
                        print(f"ストックシート確認エラー: {e}")
                
                # 家計資産シートの確認
                if '家計資産' in all_sheets:
                    try:
                        workbook_check = load_workbook(file_path)
                        worksheet_check = workbook_check['家計資産']
                        h14_value = worksheet_check['H14'].value
                        h16_value = worksheet_check['H16'].value
                        workbook_check.close()
                        
                        print(f"\n家計資産シート更新後:")
                        print(f"H14（実行日）: {h14_value}")
                        print(f"H16（実行日）: {h16_value}")
                    except Exception as e:
                        print(f"家計資産シート確認エラー: {e}")
        else:
            print(f"ファイルが見つかりません: {file_path}")
