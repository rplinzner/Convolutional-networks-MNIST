from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class DataService:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.model_dir = dir_path + 'models\\'
        self.results_dir = dir_path + 'results\\'
        self.charts_dir = dir_path + 'charts\\'
        self.photos_dir = dir_path + 'failed_photos\\'

        Path(self.dir_path).mkdir(parents=True, exist_ok=True)  # root
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)  # models
        Path(self.charts_dir).mkdir(parents=True, exist_ok=True)  # models
        Path(self.photos_dir).mkdir(parents=True,
                                    exist_ok=True)  # photos failed to recognize

        self.train_results = open(self.dir_path + 'train_results.txt', 'w')
        self.test_results = open(self.dir_path + 'test_results.txt', 'w')

        self.train_losses = []

    def __del__(self):
        self.train_results.close()
        self.test_results.close()

    def saveLineToTrainFile(self, line):
        self.train_results.write(line + "\n")

    def saveLineToTestFile(self, line):
        self.test_results.write(line + "\n")

    def saveTrainingStats(self, epoch, total_loss, avg_loss, correct, all, train_time, con_matrix):
        self.saveLineToTrainFile(f"\n")

        self.saveLineToTrainFile(f"Epoch: {epoch+1}")
        self.saveLineToTrainFile(f"Total loss: {total_loss}")
        self.saveLineToTrainFile(f"Avg loss: {avg_loss}")
        self.saveLineToTrainFile(
            f"Accuracy: {correct}/{all} ({ 100. * correct / all})%")
        self.saveLineToTrainFile(
            f"Time: {train_time // 60}min {train_time % 60}sek")

        self.saveLineToTrainFile('--- Confusion matrix ---')
        self.train_results.write('\n'.join('{}'.format(k)
                                           for k in con_matrix.tolist()))
        self.saveLineToTrainFile(f"\n")

    def saveTestingStats(self, total_loss, avg_loss, correct, all, train_time, con_matrix, epoch):
        self.saveLineToTestFile(f"\n")
        self.saveLineToTestFile(f"Epoch: {epoch+1}")
        self.saveLineToTestFile(f"Total loss: {total_loss}")
        self.saveLineToTestFile(f"Avg loss: {avg_loss}")
        self.saveLineToTestFile(
            f"Accuracy: {correct}/{all} ({ 100. * correct / all})%")
        self.saveLineToTestFile(
            f"Time: {train_time // 60}min {train_time % 60}sek")

        self.saveLineToTestFile('--- Confusion matrix ---')
        self.test_results.write('\n'.join('{}'.format(k)
                                          for k in con_matrix.tolist()))
        self.saveLineToTestFile(f"\n")

    def generateTrainChart(self, interval):
        plt.title('Training average loss')
        plt.xlabel('Epoch')
        plt.ylabel('Avg loss')
        plt.plot(np.arange(1, len(self.train_losses) * interval, interval), self.train_losses, label="Training loss")
        plt.xticks(np.arange(1, len(self.train_losses) * interval, interval))
        print( len(self.train_losses) * interval)
        print(self.train_losses)
        plt.legend()
        plt.savefig(self.charts_dir + 'plot.png')
        plt.show()
