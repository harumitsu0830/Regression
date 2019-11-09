#Support Vector Regressionの実行関数
#ハイパーパラメータ調整⇒性能評価（r2_scoreと残差プロット)

def my_SVR(DF,TARGET,Xs):
    
    '''
    Xs:説明変数リスト
    from sklearn.svm import SVR
    '''

    #########ハイパーパラメータの調整#########
    #データ
    DF_ = DF

    X, y = DF_[Xs].values, DF_[TARGET].values

    # 訓練、テスト分割
    # ホールドアウト法
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train_2, X_val, y_train_2, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    # 標準化
    scaler = StandardScaler()
    X_train_2 = scaler.fit_transform(X_train_2)
    X_val = scaler.transform(X_val)

    # optunaでハイパーパラメータを決める
    # 目的関数
    def objective(trial):
        # C
        C = trial.suggest_loguniform('C', 0.5, 2)
        # SVR
        svr = SVR(gamma='auto',C=C)
        svr.fit(X_train, y_train)
        # 予測
        y_pred = svr.predict(X_val)
        # CrossvalidationのMSEで比較（最大化がまだサポートされていない）
        return mean_squared_error(y_val, y_pred)

    # optuna
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    # 最適解
    print('study.best_params',study.best_params)
    print('study.best_value',study.best_value)
    print('study.best_trial',study.best_trial)

    #SVRのパラメータ名をint型にする
    params=study.best_params.copy()

    #########性能評価#########
    #trainとtestそれぞれでr2_scoreを計算
    
    X, y = DF_[Xs].values, DF_[TARGET].values
    # 訓練、テスト分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # 標準化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SVR
    svr = SVR(gamma='auto',**params)
    svr.fit(X_train, y_train)
    y_pred_train = svr.predict(X_train)
    y_pred_test = svr.predict(X_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    print('性能評価')
    print("トレーニングデータ　R2 : ", train_r2)
    print("テストデータ　R2 : ", test_r2)

    #残差プロット
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(20,10))
    plt.scatter(y_pred_train, y_pred_train - y_train, c='blue', marker='o', label='Training data')
    plt.scatter(y_pred_test, y_pred_test - y_test, c='lightgreen', marker='s', label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='upper left')
    plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='red')
    plt.xlim([min(y_train)-(max(y_train)-min(y_train))*0.5, max(y_train)+(max(y_train)-min(y_train))*0.5])
    plt.title('残差プロット')
    plt.show
