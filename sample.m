clear

%% データの読み込み
load ytc_matlab.mat

%% データの情報

fprintf("学習データの画像セット数: %d\n", size(X_train, 2));
fprintf("テストデータの画像セット数: %d\n", size(X_test, 2));
fprintf("特徴量の次元: %d\n", size(X_train{1}, 1));


%% 画像の表示
colormap(gray)
imagesc(reshape(X_train{1}(:, 1), 20, 20)')

