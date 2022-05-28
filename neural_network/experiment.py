from multiprocessing import Process, Manager

from neural_network import *


def cross_validation(
        input_data,
        correct_data,
        div: int):

    process_list = []
    manager = Manager()
    returned_dict = manager.dict()

    for i in range(div):
        # 訓練データとテストデータの振り分け
        idx = list(range(i * int(DATA_NUM / div), (i + 1) * int(DATA_NUM / div)))
        idx_b = np.ones(DATA_NUM, dtype=bool)
        idx_b[idx] = False
        training_idata = input_data[idx_b, :]
        training_cdata = correct_data[idx_b, :]

        test_idata = input_data[idx[0]:idx[-1] + 1, :]
        test_cdata = correct_data[idx[0]:idx[-1] + 1, :]

        process = Process(
            target=train_and_test,
            kwargs={
                'training_idata': training_idata,
                'training_cdata': training_cdata,
                'test_idata': test_idata,
                'test_cdata': test_cdata,
                'returned_dict': returned_dict,
                'process_index': i,
            }
        )

        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()

    loss_mse_list = [_['mse'] for _ in returned_dict.values()]
    loss_mae_list = [_['mae'] for _ in returned_dict.values()]

    print(f'cross validation result mse:{sum(loss_mse_list) / div} mae:{sum(loss_mae_list) / div}')


def main():
    # データの準備
    N = 1
    input_data = np.concatenate([np.random.rand(DATA_NUM, 2) * 2 * N - 1 * N, np.ones((DATA_NUM, 1), dtype=float)],
                                axis=1)
    correct_data = np.array([[my_func(_[0], _[1])] for _ in input_data])

    # 交差検証
    cross_validation(input_data, correct_data, DIV)


if __name__ == '__main__':
    main()
