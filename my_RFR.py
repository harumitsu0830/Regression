#RandomForestRegressorの実行関
#ハイパーパラメータ調整⇒性能評価（r2_scoreと残差プロット）⇒特徴量の重要度ランキングを実行。

def my_RFR(DF,TARGET,Xs):
    
    '''
    Xs:説明変数リスト
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
        n_estimators = trial.suggest_int("n_estimators", 5, 20)
        max_depth = trial.suggest_discrete_uniform("max_depth", 2, 5,1)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        random_state = 0
        # RFR
        RFR = RandomForestRegressor(n_estimators = n_estimators, 
                                    max_depth = max_depth,
                                    min_samples_leaf = min_samples_leaf,
                                   random_state = random_state)
        RFR.fit(X_train_2, y_train_2)
        # 予測
        y_pred = RFR.predict(X_val)
        # CrossvalidationのMSEで比較（最大化がまだサポートされていない）
        return mean_squared_error(y_val, y_pred)

    # optuna
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    # 最適解
    print('study.best_params',study.best_params)
    print('study.best_value',study.best_value)
    print('study.best_trial',study.best_trial)

    #RFRのパラメータ名をint型にする
    params=study.best_params.copy()
    for _k in['n_estimators','max_depth','min_samples_leaf']: params[_k] = int(params[_k])
    print('best_params',params)

    #########性能評価#########
    #trainとtestそれぞれでr2_scoreを計算
    
    X, y = DF_[Xs].values, DF_[TARGET].values
    # 訓練、テスト分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # RFR
    RFR = RandomForestRegressor(random_state = 0,
                               **params)

    RFR.fit(X_train, y_train)
    y_pred_train = RFR.predict(X_train)
    y_pred_test = RFR.predict(X_test)

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

    #########説明変数の重要度ランキング#########
    feat_labels = Xs

    # 特徴量の重要度
    importances = RFR.feature_importances_
    # 重要度の降順で特徴量のインデックスを抽出
    indices = np.argsort(importances)[::-1]
    #重要度の降順で特徴量の名称、重要度を表示
    print('特徴量の重要度ランキング')
    for f in range(X_train.shape[1]):
        print('%2d) %-*s %.3f' % (f+1,30,feat_labels[indices[f]], importances[indices[f]]))

    # 特徴量の重要度を視覚化
    f_names = []
    f_importances = []

    for i, feat in enumerate(feat_labels):
        f_names.append(feat)
        f_importances.append(importances[i])

    df_feature_importances = pd.DataFrame({'feature' : f_names, 'feature_importances' : f_importances})\
                                .sort_values('feature_importances', ascending=False)

    df_feature_importances_top10 = df_feature_importances.head(10)


    plt.style.use('fivethirtyeight')
    #グラフサイズ
    plt.figure(figsize=(10,10))
    # フォント指定（日本語文字化け防止）
    plt.rcParams['font.family'] = 'IPAPGothic' 

    plt.barh(range(10,0,-1),df_feature_importances_top10['feature_importances'].iloc[0:10], color='lightblue', align='center')
    plt.yticks(range(10,0,-1), df_feature_importances_top10['feature'], rotation=0)

    plt.title('パラメータの重要度')
    plt.xlabel('パラメータの重要度')
    plt.ylabel('パラメータ名')
    plt.show()
