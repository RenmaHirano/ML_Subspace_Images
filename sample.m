clear

%% �f�[�^�̓ǂݍ���
load ytc_matlab.mat

%% �f�[�^�̏��

fprintf("�w�K�f�[�^�̉摜�Z�b�g��: %d\n", size(X_train, 2));
fprintf("�e�X�g�f�[�^�̉摜�Z�b�g��: %d\n", size(X_test, 2));
fprintf("�����ʂ̎���: %d\n", size(X_train{1}, 1));


%% �摜�̕\��
colormap(gray)
imagesc(reshape(X_train{1}(:, 1), 20, 20)')

