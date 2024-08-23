# %%
import json
import math
import copy
import sys


def compute_metrics(data):
    cpy = copy.deepcopy(data)
    test_metrics = cpy['fc_metrics_test']
    ood_metrics = cpy['fc_metrics_test_ood']

    all_metrics = {}
    all_metrics["ML"] = test_metrics["test"]["ML"]["MSE_normalized"]
    all_metrics["ML"]["pressure_surfacic"] = test_metrics["test"]["ML"]["MSE_normalized_surfacic"]["pressure"]

    tmp = test_metrics["test"]["Physics"]
    tmp.pop("std_relative_lift");
    tmp.pop("std_relative_drag")
    all_metrics["Physics"] = tmp

    ood_dict = ood_metrics["test_ood"]["ML"]["MSE_normalized"]

    ood_dict.update({"pressure_surfacic": ood_metrics["test_ood"]["ML"]["MSE_normalized_surfacic"]["pressure"]})

    tmp = ood_metrics["test_ood"]["Physics"]
    tmp.pop("std_relative_lift");
    tmp.pop("std_relative_drag")
    ood_dict.update(tmp)

    all_metrics["OOD"] = ood_dict

    return all_metrics


def compute_speedup(data):
    speedup = {
        "ML": 1500 / data["test_mean_simulation_time"],
        "OOD": 1500 / data["test_ood_mean_simulation_time"]
    }

    return speedup


def SpeedMetric(speedUp, speedMax):
    return max(min(math.log10(speedUp) / math.log10(speedMax), 1), 0)


def compute_score(data, evaluation_path=None):
    allmetrics = compute_metrics(data)
    speedUp = compute_speedup(data)

    thresholds = {"x-velocity": (0.1, 0.2, "min"),
                  "y-velocity": (0.1, 0.2, "min"),
                  "pressure": (0.02, 0.1, "min"),
                  "pressure_surfacic": (0.08, 0.2, "min"),
                  "turbulent_viscosity": (0.5, 1.0, "min"),
                  "mean_relative_drag": (1.0, 10.0, "min"),
                  "mean_relative_lift": (0.2, 0.5, "min"),
                  "spearman_correlation_drag": (0.5, 0.8, "max"),
                  "spearman_correlation_lift": (0.94, 0.98, "max")
                  }

    configuration = {
        "coefficients": {"ML": 0.4, "OOD": 0.3, "Physics": 0.3},
        "ratioRelevance": {"Speed-up": 0.25, "Accuracy": 0.75},
        "valueByColor": {"g": 2, "o": 1, "r": 0},
        "maxSpeedRatioAllowed": 10000
    }

    accuracyResults = dict()
    for subcategoryName, subcategoryVal in allmetrics.items():
        accuracyResults[subcategoryName] = []
        for variableName, variableError in subcategoryVal.items():
            thresholdMin, thresholdMax, evalType = thresholds[variableName]
            if evalType == "min":
                if variableError < thresholdMin:
                    accuracyEval = "g"
                elif thresholdMin < variableError < thresholdMax:
                    accuracyEval = "o"
                else:
                    accuracyEval = "r"
            elif evalType == "max":
                if variableError < thresholdMin:
                    accuracyEval = "r"
                elif thresholdMin < variableError < thresholdMax:
                    accuracyEval = "o"
                else:
                    accuracyEval = "g"

            accuracyResults[subcategoryName].append(accuracyEval)

    coefficients = configuration["coefficients"]
    ratioRelevance = configuration["ratioRelevance"]
    valueByColor = configuration["valueByColor"]
    maxSpeedRatioAllowed = configuration["maxSpeedRatioAllowed"]

    mlSubscore = 0

    # Compute accuracy
    accuracyMaxPoints = ratioRelevance["Accuracy"]
    accuracyResult = sum([valueByColor[color] for color in accuracyResults["ML"]])
    accuracyResult = accuracyResult * accuracyMaxPoints / (len(accuracyResults["ML"]) * max(valueByColor.values()))
    mlSubscore += accuracyResult

    # Compute speed-up
    speedUpMaxPoints = ratioRelevance["Speed-up"]
    speedUpResult = SpeedMetric(speedUp=speedUp["ML"], speedMax=maxSpeedRatioAllowed)
    speedUpResult = speedUpResult * speedUpMaxPoints
    mlSubscore += speedUpResult

    # Compute accuracy
    accuracyResult = sum([valueByColor[color] for color in accuracyResults["Physics"]])
    accuracyResult = accuracyResult / (len(accuracyResults["Physics"]) * max(valueByColor.values()))
    physicsSubscore = accuracyResult

    oodSubscore = 0

    # Compute accuracy
    accuracyMaxPoints = ratioRelevance["Accuracy"]
    accuracyResult = sum([valueByColor[color] for color in accuracyResults["OOD"]])
    accuracyResult = accuracyResult * accuracyMaxPoints / (len(accuracyResults["OOD"]) * max(valueByColor.values()))
    oodSubscore += accuracyResult

    # Compute speed-up
    speedUpMaxPoints = ratioRelevance["Speed-up"]
    speedUpResult = SpeedMetric(speedUp=speedUp["OOD"], speedMax=maxSpeedRatioAllowed)
    speedUpResult = speedUpResult * speedUpMaxPoints
    oodSubscore += speedUpResult

    globalScore = 100 * (coefficients["ML"] * mlSubscore + coefficients["Physics"] * physicsSubscore + coefficients[
        "OOD"] * oodSubscore)

    gbl_score = {"global_score": globalScore}
    ml_score = {"ML": mlSubscore, "Physics": physicsSubscore, "OOD": oodSubscore}
    other_scores = {"ML": {"accuracy": accuracyResults["ML"], "speedup": speedUp["ML"]},
                    "Physics": {"accuracy": accuracyResults["Physics"]},
                    "OOD": {"accuracy": accuracyResults["OOD"], "speedup": speedUp["OOD"]}}

    if evaluation_path is not None:
        with open(evaluation_path, 'w') as f:
            json.dump(gbl_score, f)
            json.dump(ml_score, f)
            json.dump(other_scores, f)

    return gbl_score, ml_score, other_scores


if __name__ == "__main__":
    metrics_file = sys.argv[1:]
    metrics = json.load(open(metrics_file[0], 'r'))

    print(compute_score(metrics, None))