from torch_kit import MachineLearningPhase, create_dataset_collection


def test_dataset() -> None:
    mnist = create_dataset_collection("MNIST")
    assert (
        abs(
            len(mnist.get_dataset_util(MachineLearningPhase.Training))
            / len(mnist.get_dataset_util(MachineLearningPhase.Validation))
            - 12
        )
        < 0.01
    )
    assert mnist.get_dataset_util().channel == 1

    cifar10 = create_dataset_collection("CIFAR10")
    assert cifar10.get_dataset_util().channel == 3
    assert len(cifar10.get_dataset_util().get_labels()) == 10
    assert abs(
        len(cifar10.get_dataset_util(MachineLearningPhase.Test))
        - len(cifar10.get_dataset_util(MachineLearningPhase.Validation))
        <= 1
    )
    print("cifar10 labels are", cifar10.get_label_names())


def test_dataset_labels() -> None:
    for name in ("MNIST", "CIFAR10"):
        dc = create_dataset_collection(name)
        assert len(dc.get_labels()) == 10
