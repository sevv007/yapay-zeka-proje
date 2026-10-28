import numpy as np
import pickle
import os
from PIL import Image
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

class BaseModel:
    """
    Abstract base class for the assignment.
    -------------------------------------
    
    """
    def __init__(self):
        self.weights = {}

    def train(self, X_train, y_train, epochs=10, learning_rate=0.01):
        """
        Main training loop.
        Args:
            X_train: Training features (Numpy Array)
            y_train: Training labels (Numpy Array)
            epochs: Number of training iterations
            learning_rate: Step size for weight updates
        """
        raise NotImplementedError("You must implement the train method.")

    def predict(self, X_test):
        """
        Inference logic.
        Args:
            X_test: Test features (Numpy Array)
        Returns:
            y_pred: Predicted class labels (Numpy Array of shape (N,))
        """
        raise NotImplementedError("You must implement the predict method.")

    def save(self, path):
        """
        Saves the trained model instance using pickle.
        Args:
            path: Destination file path (e.g., 'model.pkl')
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved successfully to {path}")

    @classmethod
    def load(cls, path):
        """
        Loads a trained model instance from a file.
        Args:
            path: Source file path
        Returns:
            The loaded model instance
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {path}")
        return model

# HELPER FUNCTIONS

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Converts 4D input to 2D array for efficient convolution.
    4D input_data:(N,C,H;W) 
    2D col:(N*out_h*out_w, C*filter_h*filter_w)
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1
    
    # Pad input
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """Inverse of im2col - reconstructs image from columns. """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    
    return img[:, :, pad:H + pad, pad:W + pad]


# LAYER CLASSES

class ConvLayer:
    """Convolutional Layer with forward and backward pass."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        
        # He initialization
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.b = np.zeros(out_channels)
        
        self.x = None
        self.col = None
        self.col_W = None
        
        self.dW = None
        self.db = None
    
    def forward(self, x):
        """Forward pass of convolution."""
        N, C, H, W = x.shape
        out_h = (H + 2*self.pad - self.kernel_size) // self.stride + 1
        out_w = (W + 2*self.pad - self.kernel_size) // self.stride + 1
        
        col = im2col(x, self.kernel_size, self.kernel_size, self.stride, self.pad)
        col_W = self.W.reshape(self.out_channels, -1).T
        
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        
        self.x = x
        self.col = col
        self.col_W = col_W
        
        return out
    
    def backward(self, dout):
        """Backward pass of convolution."""
        N, C, H, W = self.x.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, self.kernel_size, self.kernel_size, self.stride, self.pad)
        
        return dx


class MaxPoolLayer:
    """Max Pooling Layer."""
    
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.x = None
        self.arg_max = None
    
    def forward(self, x):
        """Forward pass of max pooling."""
        N, C, H, W = x.shape
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1
        
        col = im2col(x, self.pool_size, self.pool_size, self.stride, 0)
        col = col.reshape(-1, self.pool_size * self.pool_size)
        
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        
        self.x = x
        self.arg_max = arg_max
        
        return out
    
    def backward(self, dout):
        """Backward pass of max pooling."""
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_size
        dmax = np.zeros((dout.size, pool_size * pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size * pool_size,))
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_size, self.pool_size, self.stride, 0)
        
        return dx


class FullyConnectedLayer:
    """Fully Connected (Dense) Layer."""
    
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        """Forward pass."""
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    
    def backward(self, dout):
        """Backward pass."""
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class ReLU:
    """ReLU Activation Function."""
    
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Softmax:
    """Softmax with Cross-Entropy Loss."""
    
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        """Forward pass with loss calculation."""
        self.t = t
        self.y = self.softmax(x)
        self.loss = self.cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self):
        """Backward pass."""
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
    
    def softmax(self, x):
        """Numerically stable softmax."""
        x = x - np.max(x, axis=1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    
    def cross_entropy_error(self, y, t):
        """Cross-entropy loss."""
        batch_size = y.shape[0]
        return -np.sum(t * np.log(y + 1e-7)) / batch_size

# MY MODEL

class MyModel(BaseModel):
    """
    CNN Implementation from Scratch
    """
    
    def __init__(self, input_shape=(3, 56, 56), num_classes=2):
        super().__init__()
        
        # Architecture parameters
        self.input_shape = input_shape
        self.num_classes = num_classes
        # Normalization parameters
        self.mean = None
        self.std = None
        # Initialize layers
        # Conv1: 3 -> 32 channels, 3x3 kernel
        self.conv1 = ConvLayer(in_channels=3, out_channels=32, kernel_size=3, stride=1, pad=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPoolLayer(pool_size=2, stride=2)  # 56x56 -> 28x28

        
        # Conv2: 32 -> 64 channels, 3x3 kernel
        self.conv2 = ConvLayer(in_channels=32, out_channels=64, kernel_size=3, stride=1, pad=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPoolLayer(pool_size=2, stride=2)  # 28x28 -> 14x14

        # Conv3: 64 -> 128 channels, 3x3 kernel
        self.conv3 = ConvLayer(in_channels=64, out_channels=128, kernel_size=3, stride=1, pad=1)
        self.relu3 = ReLU()
        self.pool3 = MaxPoolLayer(pool_size=2, stride=2)  # 14x14 -> 7x7
        
        # Fully connected layers
        # After 3 pooling layers: 56 -> 28 -> 14 -> 7
        fc_input_size = 128 * 7 * 7 
        self.fc1 = FullyConnectedLayer(fc_input_size, 256)
        self.relu4 = ReLU()
        self.dropout_rate = 0.5
        self.dropout_mask = None
        self.fc2 = FullyConnectedLayer(256, num_classes)
        
        # Softmax + Loss
        self.softmax = Softmax()
        
        # For tracking
        self.layers = [self.conv1, self.relu1, self.pool1, 
                       self.conv2, self.relu2, self.pool2,
                       self.conv3, self.relu3, self.pool3,
                       self.fc1, self.relu4, self.fc2]
    
    def forward(self, X, training=True):
        """Forward pass through the network."""
        # Conv block 1
        h = self.conv1.forward(X)
        h = self.relu1.forward(h)
        h = self.pool1.forward(h)
        
        # Conv block 2
        h = self.conv2.forward(h)
        h = self.relu2.forward(h)
        h = self.pool2.forward(h)
        
        # Conv block 3
        h = self.conv3.forward(h)
        h = self.relu3.forward(h)
        h = self.pool3.forward(h)
        
        # Flatten
        h = h.reshape(h.shape[0], -1)
        
        # Fully connected with dropout
        h = self.fc1.forward(h)
        h = self.relu4.forward(h)
        
        # Dropout
        if training:
            self.dropout_mask = np.random.rand(*h.shape) > self.dropout_rate
            h = h * self.dropout_mask
        else:
            h = h * (1 - self.dropout_rate)
        
        h = self.fc2.forward(h)
        
        return h
    
    def backward(self, dout):
        """Backward pass through the network."""
        # FC layers
        dout = self.fc2.backward(dout)
        
        # Dropout backward
        if self.dropout_mask is not None:
            dout = dout * self.dropout_mask
        
        dout = self.relu4.backward(dout)
        dout = self.fc1.backward(dout)
        
        # Reshape for conv layers
        dout = dout.reshape(dout.shape[0], 128, 7, 7)
        
        # Conv block 3
        dout = self.pool3.backward(dout)
        dout = self.relu3.backward(dout)
        dout = self.conv3.backward(dout)
        
        # Conv block 2
        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.conv2.backward(dout)
        
        # Conv block 1
        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.conv1.backward(dout)
        
        return dout
    
    def update_weights(self, learning_rate):
        """Update all weights using gradients."""
        l2_lambda = 0.0001
        # Conv layers
        self.conv1.W -= learning_rate * (self.conv1.dW + l2_lambda * self.conv1.W)
        self.conv1.b -= learning_rate * self.conv1.db
        
        self.conv2.W -= learning_rate * (self.conv2.dW + l2_lambda * self.conv2.W)
        self.conv2.b -= learning_rate * self.conv2.db
        
        self.conv3.W -= learning_rate * (self.conv3.dW + l2_lambda * self.conv3.W)
        self.conv3.b -= learning_rate * self.conv3.db
        
        # FC layers
        self.fc1.W -= learning_rate * (self.fc1.dW + l2_lambda * self.fc1.W)
        self.fc1.b -= learning_rate * self.fc1.db
        
        self.fc2.W -= learning_rate * (self.fc2.dW + l2_lambda * self.fc2.W)
        self.fc2.b -= learning_rate * self.fc2.db

    def normalize(self, X):
        """Normalize data using stored mean and std."""
        if self.mean is not None and self.std is not None:
            return (X - self.mean) / self.std
        return X
    
    
    def train(self, X_train, y_train, epochs=10, learning_rate=0.01, batch_size=32, X_val=None, y_val=None):
        """
        Training loop with mini-batch gradient descent and learning rate decay.
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Train set size: {len(X_train)}, Batch size: {batch_size}")
        
        # Convert to channel-first format: (N, H, W, C) -> (N, C, H, W)
        if X_train.shape[-1] == 3:
            X_train = X_train.transpose(0, 3, 1, 2)
        
        if X_val is not None and X_val.shape[-1] == 3:
            X_val = X_val.transpose(0, 3, 1, 2)
        
        # Normalize data (mean subtraction and std normalization)
        self.mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
        self.std = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-7
        X_train = self.normalize(X_train)
        if X_val is not None:
            X_val = self.normalize(X_val)
        
        # Convert labels to one-hot
        y_train_onehot = self._to_onehot(y_train)
        
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size
        
        train_losses = []
        val_accuracies = []
        best_val_acc = 0
        patience = 8
        patience_counter = 0
        
        initial_lr = learning_rate
        
        for epoch in range(epochs):
            # Learning rate decay
            current_lr = initial_lr * (0.95 ** epoch)
            
            epoch_loss = 0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_onehot_shuffled = y_train_onehot[indices]
            
            # Mini-batch training
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_onehot_shuffled[start_idx:end_idx]
                
                # Forward pass (training mode)
                scores = self.forward(X_batch, training=True)
                
                # Compute loss
                loss = self.softmax.forward(scores, y_batch)
                epoch_loss += loss
                
                # Backward pass
                dout = self.softmax.backward()
                self.backward(dout)
                
                # Update weights
                self.update_weights(current_lr)
            
            avg_loss = epoch_loss / n_batches
            train_losses.append(avg_loss)
            
            # Validation accuracy
            val_acc = 0
            if X_val is not None and y_val is not None:
                val_acc = self.evaluate(X_val, y_val)
                val_accuracies.append(val_acc)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f} - LR: {current_lr:.6f}")
                
                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")
        
        print("Training completed.")
        return train_losses, val_accuracies
    
    
    def predict(self, X_test, batch_size=32):
        """
        Inference on test data with batching.
        
        """
        n_samples = X_test.shape[0]
        y_pred = np.zeros(n_samples, dtype=int)
        
        # Process in batches
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X_test[i:end_idx]
            
            # Convert to channel-first if needed
            if X_batch.shape[-1] == 3:
                X_batch = X_batch.transpose(0, 3, 1, 2)

            X_batch = self.normalize(X_batch)
            scores = self.forward(X_batch, training=False)  
            y_pred[i:end_idx] = np.argmax(scores, axis=1)
        
        return y_pred
    
    def evaluate(self, X, y, batch_size=32):
        """Evaluate accuracy on given data with batching to avoid memory issues.
        
        """
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples, dtype=int)
        
        # Process in batches
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X[i:end_idx]
            
            # Convert to channel-first if needed
            if X_batch.shape[-1] == 3:
                X_batch = X_batch.transpose(0, 3, 1, 2)
            
            X_batch = self.normalize(X_batch)
            scores = self.forward(X_batch, training=False)  
            y_pred[i:end_idx] = np.argmax(scores, axis=1)
        
        accuracy = np.mean(y_pred == y)
        return accuracy
    
    def _to_onehot(self, y):
        """Convert labels to one-hot encoding."""
        n_samples = y.shape[0]
        onehot = np.zeros((n_samples, self.num_classes))
        onehot[np.arange(n_samples), y] = 1
        return onehot


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

def load_images_from_folder(folder_path, target_size=(56, 56)):
    """
    Load all images from a folder.
    """
    images = []
    folder = Path(folder_path)

    # Klasörün mevcut olup olmadığını kontrol et
    if not folder.exists():
        raise FileNotFoundError(
            f"Klasör bulunamadı: {folder.resolve()}"
        )

    # JPG, JPEG ve PNG uzantılı resimleri bul
    image_paths = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in
        (".jpg", ".jpeg", ".png")
    )

    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img = img.resize(target_size, Image.BILINEAR)
                img_array = np.array(img, dtype=np.float32) / 255.0
                images.append(img_array)
        except Exception as e:
            print(f"Resim okunamadı: {img_path} - {e}")

    # Klasör boşsa anlaşılır bir hata göster
    if not images:
        raise ValueError(
            f"Bu klasörde okunabilir resim bulunamadı: "
            f"{folder.resolve()}"
        )

    
    return np.array(images)

def load_dataset(dataset_path, target_size=(56, 56)):
    """
    Load the complete dataset.
    """
    dataset_path = Path(dataset_path)
    
    print("Loading training data...")
    # Load train data
    train_cat = load_images_from_folder(dataset_path / 'train' / 'cat', target_size)
    train_bicycle = load_images_from_folder(dataset_path / 'train' / 'bicycle', target_size)
    
    X_train = np.concatenate([train_cat, train_bicycle], axis=0)
    y_train = np.concatenate([np.zeros(len(train_cat)), np.ones(len(train_bicycle))]).astype(int)
    
    print(f"Train: {len(train_cat)} cats, {len(train_bicycle)} bicycles")
    
    print("Loading validation data...")
    # Load validation data
    val_cat = load_images_from_folder(dataset_path / 'val' / 'cat', target_size)
    val_bicycle = load_images_from_folder(dataset_path / 'val' / 'bicycle', target_size)
    
    X_val = np.concatenate([val_cat, val_bicycle], axis=0)
    y_val = np.concatenate([np.zeros(len(val_cat)), np.ones(len(val_bicycle))]).astype(int)
    
    print(f"Val: {len(val_cat)} cats, {len(val_bicycle)} bicycles")
    
    return X_train, y_train, X_val, y_val


def compute_confusion_matrix(y_true, y_pred, num_classes=2):
    """
    Compute confusion matrix.
    """
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        confusion_matrix[y_true[i], y_pred[i]] += 1
    return confusion_matrix


# MAIN EXECUTION

if __name__ == "__main__":
    """
    Main execution script for training and saving the model.
    """
    
    # Configuration
    DATASET_PATH = "MP3_Dataset" 
    
    # Hyperparameters
    EPOCHS = 20
    LEARNING_RATE = 0.005
    BATCH_SIZE = 16
    TARGET_SIZE = (56, 56)
    
    print("="*60)
    print("CNN FROM SCRATCH - TRAINING SCRIPT")
    print("="*60)
    
    # Load dataset
    X_train, y_train, X_val, y_val = load_dataset(DATASET_PATH, target_size=TARGET_SIZE)
    
    print(f"\nDataset loaded successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    
    # Initialize model
    print("\nInitializing model...")
    model = MyModel(input_shape=(3, TARGET_SIZE[0], TARGET_SIZE[1]), num_classes=2)
    
    # Train model
    print("\n" + "="*60)
    train_losses, val_accuracies = model.train(
        X_train, y_train,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        X_val=X_val,
        y_val=y_val
    )
    print("="*60)
    
    # Final evaluation
    print("\nFinal Evaluation:")
    train_acc = model.evaluate(X_train, y_train)
    val_acc = model.evaluate(X_val, y_val)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # Save model
    model_path = f"file.pkl"
    model.save(model_path)
    
    print(f"\nModel saved to {model_path}")
    print("\nTraining complete!")
    
    # Test loading
    print("\nTesting model loading...")
    loaded_model = MyModel.load(model_path)
    test_acc = loaded_model.evaluate(X_val, y_val)
    print(f"Loaded model validation accuracy: {test_acc:.4f}")
    
    # Save training history for report
    np.save(f"train_losses.npy", train_losses)
    np.save(f"val_accuracies.npy", val_accuracies)
    print(f"\nTraining history saved for report generation")
    
    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    y_pred = model.predict(X_val)
    confusion_mat = compute_confusion_matrix(y_val, y_pred, num_classes=2)
    np.save(f"confusion_matrix.npy", confusion_mat)
    print("Confusion Matrix:")
    print(confusion_mat)
    print(f"Confusion matrix saved to confusion_matrix.npy")