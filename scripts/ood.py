import argparse
from scr.data import get_dataset
from scr.generative_models import get_model_from_file
from sklearn.metrics import roc_auc_score
import datetime

def calc_log_prob(
    model_class,
    model_instance,
    refit_encoder,
    score_method,
    train_dataloader,
    test_dataloader,
):
    if refit_encoder:
        model_class.refit_encoder(model_instance, train_dataloader)

    log_probs_train = model_class.calc_score(model_instance, score_method, train_dataloader)
    log_probs_test = model_class.calc_score(model_instance, score_method, test_dataloader)

    return log_probs_train, log_probs_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weight_file")
    parser.add_argument("score_method")
    args = parser.parse_args()

    model, model_class = get_model_from_file(args.weight_file)
    dataset = args.weight_file.split("_")[0]

    all_datasets = ['mnist', 'fmnist', 'kmnist', 'omniglot']
    res = {}
    for dataset_name in all_datasets:
        train, val, test = get_dataset(dataset_name)
        train_dataloader = torch.utils.data.DataLoader(train, batch_size=64)
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=64)
        test_dataloader = torch.utils.data.DataLoader(test, batch_size=64)

        res[f"train_{dataset_name}"], res[f"test_{dataset_name}"] = calc_log_prob(
            model_class, model, True, args.score_method, train_dataloader, test_dataloader
        )

        res[f"train_{dataset_name}_refit"], res[f"test_{dataset_name}_refit"] = calc_log_prob(
            model_class, model, False, args.score_method, train_dataloader, test_dataloader
        )

    # calculate auroc score
    os.makedirs("results/", exist_ok=True)
    with open("results/ood_scores.txt", "w")
        file.write("primary, secondary, refit, auroc")

    for refit in [False, True]:
        primary = res[f"train_{dataset}_refit"] if refit else res[f"train_{dataset}"]
        for dataset_name in all_datasets if not dataset_name == dataset:
            secondary = res[f"train_{dataset_name}_refit"] if refit else res[f"train_{dataset_name}"]
            labels = torch.cat(torch.zeros(len(seconday)), torch.ones(len(primary)))
            score = roc_auc_score(torch.cat(secondary, primary))
            
            with open("results/ood_scores.txt", "a"):
                file.write(
                    f"{dataset}, {dataset_name}, {refit}, {score}"
                )
