from datetime import datetime
from multiprocessing import Process, Manager

from neural_network_summit import *


def cross_validation(
        input_data,
        correct_data,
        div: int,
        *args,
        **kwargs,
):
    """交差検証"""
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

        kwargs_dict = {
                'training_idata': training_idata,
                'training_cdata': training_cdata,
                'test_idata': test_idata,
                'test_cdata': test_cdata,
                'returned_dict': returned_dict,
                'process_index': i,
                **kwargs
            }

        process = Process(
            target=train_and_test,
            kwargs=kwargs_dict
        )

        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()

    loss_list = [_['sse'] for _ in returned_dict.values()]

    print(f'cross validation result:{sum(loss_list) / div}')

    return sum(loss_list) / div


def single_experiment(
        input_data,
        correct_data,
        div
):
    """1回のみの実験"""
    i = 0
    # 訓練データとテストデータの振り分け
    idx = list(range(i * int(DATA_NUM / div), (i + 1) * int(DATA_NUM / div)))
    idx_b = np.ones(DATA_NUM, dtype=bool)
    idx_b[idx] = False
    training_idata = input_data[idx_b, :]
    training_cdata = correct_data[idx_b, :]

    test_idata = input_data[idx[0]:idx[-1] + 1, :]
    test_cdata = correct_data[idx[0]:idx[-1] + 1, :]

    train_and_test(training_idata, training_cdata, test_idata, test_cdata, {}, 0)


def half_nodes_change(half_nodes: List[int],
                      *args,
                      **kwargs):
    """中間層数を変えて実験"""
    now = datetime.now()
    path = f"result/{now.strftime('%m-%d-%H:%M:%S')}.csv"
    with open(path, mode="w") as f:
        f.write('half_nodes,loss_sse\n')

    for i in half_nodes:
        loss = cross_validation(half_nodes=i, *args, **kwargs)
        with open(path, mode="a") as f:
            f.write(f'{i},{loss}\n')


def main():
    # データの準備
    N = 1
    input_data = np.concatenate([np.random.rand(DATA_NUM, 2) * 2 * N - 1 * N, np.ones((DATA_NUM, 1), dtype=float)],
                                axis=1)
    correct_data = np.array([[my_func(_[0], _[1])] for _ in input_data])

    # 交差検証
    # cross_validation(input_data, correct_data, DIV)
    # single_experiment(input_data, correct_data, DIV)

    # 実験
    half_nodes = [10, 100, 1000, 3000]
    half_nodes_change(half_nodes, input_data, correct_data, DIV)


if __name__ == '__main__':
    main()
