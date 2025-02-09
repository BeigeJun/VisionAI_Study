import os
import torch
import matplotlib.pyplot as plt


class Draw_Graph():
    def __init__(self, save_path, patience):
        self.patience = patience
        self.save_path = save_path

        self.train_losses = []
        self.train_accuracies = []

        self.val_losses = []
        self.val_accuracies = []

        self.top_accuracy_train = 0
        self.top_accuracy_validation = 0
        self.top_accuracy_train_epoch = 0
        self.top_accuracy_validation_epoch = 0

        self.bottom_loss_train = float('inf')
        self.bottom_loss_validation = float('inf')
        self.bottom_loss_train_epoch = 0
        self.bottom_loss_validation_epoch = 0

        os.makedirs(save_path, exist_ok=True)

        plt.ion()
        self.fig, self.axs = plt.subplots(1, 2, figsize=(16, 4))

    def save_plt(self):
        plt.savefig(self.save_path + '//training_validation_graphs.png')
        plt.close()

    def save_model(self, model, save_type='Best_Accuracy_Train_MLP'):
        model_path_make = self.save_path + '//' + save_type + '.pth'
        torch.save(model.state_dict(), model_path_make)

    def append_train_losses_and_acc(self, loss, acc):
        self.train_losses.append(loss)
        self.train_accuracies.append(acc)

    def update_train_losses_or_acc(self, updated_item, epoch, choose_mode="loss"):
        if choose_mode == "loss":
            self.top_accuracy_train = updated_item
            self.top_accuracy_train_epoch = epoch
        elif choose_mode == "acc":
            self.bottom_loss_train = updated_item
            self.bottom_loss_train_epoch = epoch

    def update_validation_losses_or_acc(self, updated_item, epoch, choose_mode="loss"):
        if choose_mode == "loss":
            self.top_accuracy_validation = updated_item
            self.top_accuracy_validation_epoch = epoch
        elif choose_mode == "acc":
            self.bottom_loss_validation = updated_item
            self.bottom_loss_validation_epoch = epoch

    def save_train_info(self, patience_count):
        plt.savefig(self.save_path + '//training_validation_graphs.png')
        with open(self.save_path + '//numbers.txt', "w") as file:
            file.write(
                f"Top Accuracy Train Epoch : {self.top_accuracy_train_epoch} Accuracy : {self.top_accuracy_train}\n"
                f"Top Accuracy Validation Epoch : {self.top_accuracy_validation_epoch} Accuracy : {self.top_accuracy_validation}\n"
                f"Bottom Loss Train Epoch : {self.bottom_loss_train_epoch} Loss : {self.bottom_loss_train}\n"
                f"Bottom Loss Validation Epoch : {self.bottom_loss_validation_epoch} Loss : {self.bottom_loss_validation}\n"
                f"Patience Count : {patience_count}/{self.patience}\n")

    def save_test_info(self, total, correct, accuracy):
        with open(self.save_path + '//Test_Result.txt', "w") as file:
            file.write(f"Total Num : {total}, Correct Num : {correct}\n"
                       f"Accuracy : {accuracy}")

    def update_graph(self, model, device, validation_loader, criterion, epoch):
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        self.val_losses.append(val_loss / len(validation_loader))
        val_accuracy = correct_val / total_val
        self.val_accuracies.append(val_accuracy)

        self.axs[0].clear()
        self.axs[1].clear()

        self.axs[0].plot(range(10, epoch + 2, 10), self.train_accuracies, label='Train Accuracy', color='red', linewidth=0.5)
        self.axs[0].plot(range(10, epoch + 2, 10), self.val_accuracies, label='Validation Accuracy', color='blue',
                    linewidth=0.5)
        self.axs[0].set_xlabel('Epochs')
        self.axs[0].set_ylabel('Accuracy')
        self.axs[0].set_title('Training and Validation Accuracy')
        self.axs[0].legend()

        self.axs[1].plot(range(10, epoch + 2, 10), self.train_losses, label='Train Loss', color='red', linewidth=0.5)
        self.axs[1].plot(range(10, epoch + 2, 10), self.val_losses, label='Validation Loss', color='blue', linewidth=0.5)
        self.axs[1].set_xlabel('Epochs')
        self.axs[1].set_ylabel('Loss')
        self.axs[1].set_title('Training and Validation Loss')
        self.axs[1].legend()

        for tick in self.axs[0].get_xticks():
            self.axs[0].axvline(x=tick, color='gray', linestyle='-', linewidth=0.1)

        for tick in self.axs[0].get_yticks():
            self.axs[0].axhline(y=tick, color='gray', linestyle='-', linewidth=0.1)

        for tick in self.axs[1].get_xticks():
            self.axs[1].axvline(x=tick, color='gray', linestyle='-', linewidth=0.1)

        for tick in self.axs[1].get_yticks():
            self.axs[1].axhline(y=tick, color='gray', linestyle='-', linewidth=0.1)

        plt.draw()
        plt.pause(0.1)
