from utils import plot_regression_scatter
from utils import plot_classification_scatter
class result_analysis:
 def __init__(self, data, result):
    self.molecules = data.molecules
    self.trues = data.labels
    self.preds = result.preds
 def visualize(self):
     plot_regression_scatter(self.trues, self.preds)
     
 def print_results(self):
     plot_classification_scatter(self.trues, self.preds)