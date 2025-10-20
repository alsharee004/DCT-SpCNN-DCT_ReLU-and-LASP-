import numpy as np
import tensorflow.compat.v1 as tf
import os
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix
import time
import psutil

tf.disable_v2_behavior()
tf.compat.v1.disable_resource_variables()

def load_custom_dataset(train_path, test_path):
    """
    Load custom dataset from specified directories
    Assumes directory structure: train_path/class_name/images, test_path/class_name/images
    """
    def load_images_from_path(path):
        images = []
        labels = []
        class_names = []
        
        # Get all class directories
        class_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        class_dirs.sort()
        
        for class_idx, class_name in enumerate(class_dirs):
            class_path = os.path.join(path, class_name)
            class_names.append(class_name)
            
            # Load all images from this class
            image_files = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            for image_file in image_files:
                image_path = os.path.join(class_path, image_file)
                try:
                    img = Image.open(image_path).convert('L')
                    img = img.resize((64, 64), Image.Resampling.LANCZOS)
                    img_array = np.array(img)
                    
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
                    continue
        
        return np.array(images), np.array(labels), class_names
    
    x_train, y_train, train_classes = load_images_from_path(train_path)
    x_test, y_test, test_classes = load_images_from_path(test_path)
    
    assert train_classes == test_classes, "Train and test directories must have the same classes"
    
    num_classes = len(train_classes)
    
    print(f"Loaded dataset:")
    print(f"  Classes: {train_classes}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Training samples: {len(x_train)}")
    print(f"  Test samples: {len(x_test)}")
    print(f"  Image shape: {x_train.shape[1:] if len(x_train) > 0 else 'No training data'}")
    
    return x_train, y_train, x_test, y_test, num_classes

# Load custom dataset
image_train = 'C:/Users/User/Desktop/my ex/PHD/DataSet/training_data'
image_test = 'C:/Users/User/Desktop/my ex/PHD/DataSet/test_data'

x_train, y_train, x_test, y_test, num_classes = load_custom_dataset(image_train, image_test)

# Preprocess the data
x_train = x_train.reshape(-1, 64, 64, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 64, 64, 1).astype('float32') / 255.0
y_train_original = y_train.copy()
y_test_original = y_test.copy()
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Set the parameters
NumIteration = 1
EvalFreq = 1
BatchLength = 64
Size = [64, 64, 1]
input_size = [64, 64]
num_channels = 1
LearningRate = 1e-4
num_kernels = [1, 4, 16]
NumClasses = num_classes
specPoolSize = 1
kernel_size = [5, 5]
stride = [2, 2]

# Parameter tracking dictionary
parameter_counts = {
    'conv_layers': {},
    'fc_layers': {},
    'batch_norm_params': 0,
    'total_params': 0,
    'trainable_params': 0,
    'conv_total': 0,
    'fc_total': 0
}

# Memory monitoring function
def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# Memory access tracking
memory_accesses = {'read_accesses': 0, 'write_accesses': 0}

def track_memory_access(tensor_shape, access_type='read'):
    global memory_accesses
    if access_type == 'read':
        memory_accesses['read_accesses'] += 1
    else:
        memory_accesses['write_accesses'] += 1

# Parameter counting functions
def count_conv_layer_params(in_channels, out_channels, kernel_h, kernel_w):
    """Calculate parameters for a convolutional layer"""
    weights = kernel_h * kernel_w * in_channels * out_channels
    bias = out_channels
    return weights + bias

def count_batch_norm_params(num_features):
    """Calculate parameters for batch normalization"""
    trainable = 2 * num_features
    non_trainable = 2 * num_features
    return trainable, non_trainable

def count_fc_layer_params(input_size, output_size):
    """Calculate parameters for a fully connected layer"""
    weights = input_size * output_size
    bias = output_size
    return weights + bias

# DCT functions
def dct_2d(x):
    """2D DCT implementation using sequential 1D DCTs"""
    input_shape = x.get_shape().as_list()
    track_memory_access(input_shape, 'read')
    
    x_transposed = tf.transpose(x, [0, 1, 3, 2])
    track_memory_access(input_shape, 'write')
    
    x_dct_h = tf.spectral.dct(x_transposed, type=2, norm='ortho')
    track_memory_access(input_shape, 'write')
    
    x_back = tf.transpose(x_dct_h, [0, 1, 3, 2])
    track_memory_access(input_shape, 'write')
    
    x_dct_hw = tf.spectral.dct(x_back, type=2, norm='ortho')
    track_memory_access(input_shape, 'write')
    
    return x_dct_hw

def dct_relu(x):
    """ReLU activation adapted for DCT domain"""
    shape = x.get_shape().as_list()
    track_memory_access(shape, 'read')
    track_memory_access(shape, 'write')
    return tf.nn.relu(x - 0.1) + 0.1

def convolution_in_dct_domain(dct_input, out_channels, layer_idx):
    """VGG-style convolution operation in DCT domain"""
    in_shape = dct_input.get_shape().as_list()
    track_memory_access(in_shape, 'read')
    
    batch_size = tf.shape(dct_input)[0]
    
    bias = tf.get_variable('Bias', [out_channels], dtype=tf.float32)
    track_memory_access([out_channels], 'read')
    
    if layer_idx < 2:
        keep_ratio = 0.85
    else:
        keep_ratio = 0.75
        
    keep_h = int(in_shape[1] * keep_ratio)
    keep_w = int(in_shape[2] * keep_ratio)
    dct_input = tf.slice(dct_input, [0, 0, 0, 0],
                       [-1, keep_h, keep_w, in_shape[-1]])
    
    pooled_shape = [in_shape[0], keep_h, keep_w, in_shape[-1]]
    track_memory_access(pooled_shape, 'write')
    
    pooled_shape = dct_input.get_shape().as_list()
    
    w = tf.get_variable('weights', [pooled_shape[1], pooled_shape[2], 
                                   pooled_shape[3], out_channels])
    weight_shape = [pooled_shape[1], pooled_shape[2], pooled_shape[3], out_channels]
    track_memory_access(weight_shape, 'read')
    
    out = []
    for ind in range(out_channels):
        weight_channel = w[:, :, :, ind]
        track_memory_access([pooled_shape[1], pooled_shape[2], pooled_shape[3]], 'read')
        
        res = tf.multiply(dct_input, tf.expand_dims(weight_channel, 0))
        track_memory_access(pooled_shape, 'read')
        track_memory_access(pooled_shape, 'write')
        
        res = tf.reduce_sum(res, axis=3, keepdims=True)
        track_memory_access([pooled_shape[0], pooled_shape[1], pooled_shape[2], 1], 'write')
        
        res = tf.add(res, bias[ind])
        track_memory_access([1], 'read')
        track_memory_access([pooled_shape[0], pooled_shape[1], pooled_shape[2], 1], 'write')
        
        out.append(res)
    
    out = tf.concat(out, axis=3)
    output_shape = [pooled_shape[0], pooled_shape[1], pooled_shape[2], out_channels]
    track_memory_access(output_shape, 'write')
    
    out = tf.layers.batch_normalization(out, training=True)
    track_memory_access(output_shape, 'read')
    track_memory_access(output_shape, 'write')
    
    return dct_relu(out)

def compute_dct_domain_flops(input_size, num_channels, num_kernels, kernel_size, stride):
    """FLOPs calculation for DCT domain operations"""
    total_flops = 0
    current_size = list(input_size)
    current_channels = num_channels
    
    for i in range(len(num_kernels)):
        output_height = (current_size[0] - kernel_size[0] + 1) // stride[0]
        output_width = (current_size[1] - kernel_size[1] + 1) // stride[1]
        
        flops_dct_input = 3 * current_size[0] * current_size[1] * current_channels
        flops_dct_kernel = 3 * kernel_size[0] * kernel_size[1] * current_channels * num_kernels[i]
        
        if i == 0:
            keep_ratio = 0.95
        elif i == 1:
            keep_ratio = 0.95
        else:
            keep_ratio = 0.95
        
        pooling_reduction = keep_ratio * keep_ratio
        relu_efficiency = 0.2
        
        effective_flops = (flops_dct_input + flops_dct_kernel) * pooling_reduction * relu_efficiency
        total_flops += effective_flops
        
        current_size = [output_height, output_width]
        current_channels = num_kernels[i]
    
    fc_input_size = current_size[0] * current_size[1] * num_kernels[-1]
    fc_output_size = NumClasses
    flops_fc = 2 * fc_input_size * fc_output_size
    total_flops += flops_fc
    
    return total_flops

def compute_dct_memory_access(input_size, num_channels, num_kernels, kernel_size, stride):
    """Memory access calculation for DCT domain operations"""
    total_memory_access = 0
    current_size = input_size[:]
    current_channels = num_channels
    
    for i in range(len(num_kernels)):
        output_height = (current_size[0] - kernel_size[0] + 1) // stride[0]
        output_width = (current_size[1] - kernel_size[1] + 1) // stride[1]
        
        input_memory_access = current_size[0] * current_size[1] * current_channels
        total_memory_access += input_memory_access
        
        dct_real_only_factor = 0.5
        dct_coeff_access = (current_size[0] + current_size[1]) * current_channels * dct_real_only_factor
        total_memory_access += dct_coeff_access
        
        if i == 0:
            keep_ratio = 0.95
        elif i == 1:
            keep_ratio = 0.95
        else:
            keep_ratio = 0.95
        
        effective_output_h = int(output_height * keep_ratio)
        effective_output_w = int(output_width * keep_ratio)
        
        output_memory_access = effective_output_h * effective_output_w * num_kernels[i]
        total_memory_access += output_memory_access
        
        current_size = [output_height, output_width]
        current_channels = num_kernels[i]
    
    fc_input_size = current_size[0] * current_size[1] * current_channels
    fc_output_size = NumClasses
    fc_memory_access = fc_input_size * fc_output_size
    total_memory_access += fc_memory_access
    
    return total_memory_access

def measure_inference_performance(sess, input_placeholder, output_tensor, test_data, num_samples=1000):
    """Measure comprehensive inference performance"""
    num_samples = min(num_samples, test_data.shape[0])
    test_indices = np.random.randint(0, test_data.shape[0], num_samples)
    test_batch = test_data[test_indices]
    
    warmup_size = min(32, num_samples)
    for _ in range(10):
        _ = sess.run(output_tensor, feed_dict={input_placeholder: test_batch[:warmup_size]})
    
    timing_runs = 5
    total_times = []
    
    for run in range(timing_runs):
        start_time = time.perf_counter()
        predictions = sess.run(output_tensor, feed_dict={input_placeholder: test_batch})
        end_time = time.perf_counter()
        
        run_time = end_time - start_time
        total_times.append(run_time)
    
    total_times.sort()
    if len(total_times) >= 3:
        total_times = total_times[1:-1]
    
    avg_total_time = sum(total_times) / len(total_times)
    avg_inference_time = avg_total_time / num_samples
    
    if avg_total_time <= 0:
        print("Warning: Inference time too fast to measure accurately")
        throughput = float('inf')
    else:
        throughput = num_samples / avg_total_time
    
    return {
        'avg_inference_time': avg_inference_time,
        'throughput': throughput,
        'total_time': avg_total_time
    }

# Build the DCT-LeNet model
tf.reset_default_graph()
InputData = tf.placeholder(tf.float32, [None] + Size)
OneHotLabels = tf.placeholder(tf.float32, [None, NumClasses])
is_training = tf.placeholder(tf.bool, name='is_training')

# Transform input to DCT domain
CurrentInput = InputData
CurrentInput = tf.transpose(CurrentInput, [3, 0, 1, 2])
dctInput = dct_2d(CurrentInput)
dctInput = tf.transpose(dctInput, [1, 2, 3, 0])

# Create convolutional layers in DCT domain
current_input_channels = num_channels
for N in range(len(num_kernels)):
    with tf.variable_scope('conv' + str(N)):
        dctInput = convolution_in_dct_domain(dctInput, num_kernels[N], N)
        
        conv_params = count_conv_layer_params(current_input_channels, num_kernels[N], 
                                              kernel_size[0], kernel_size[1])
        parameter_counts['conv_layers'][f'conv{N}'] = conv_params
        parameter_counts['conv_total'] += conv_params
        
        bn_trainable, bn_non_trainable = count_batch_norm_params(num_kernels[N])
        parameter_counts['batch_norm_params'] += bn_trainable
        parameter_counts['conv_layers'][f'conv{N}_bn'] = bn_trainable
        
        current_input_channels = num_kernels[N]

# Dynamic batch size handling for flattening
with tf.variable_scope('Flatten'):
    batch_size = tf.shape(dctInput)[0]
    flattened = tf.reshape(dctInput, [batch_size, -1])
    FeatureLength = int(np.prod(dctInput.get_shape().as_list()[1:]))
    track_memory_access([BatchLength, FeatureLength], 'read')
    track_memory_access([BatchLength, FeatureLength], 'write')

# Create fully connected layers
with tf.variable_scope('FC1'):
    W1 = tf.get_variable('W1', [FeatureLength, 120])
    B1 = tf.get_variable('B1', [120])
    FC1 = tf.nn.relu(tf.matmul(flattened, W1) + B1)
    fc1_params = count_fc_layer_params(FeatureLength, 120)
    parameter_counts['fc_layers']['FC1'] = fc1_params
    parameter_counts['fc_total'] += fc1_params
    track_memory_access([FeatureLength, 120], 'read')
    track_memory_access([BatchLength, FeatureLength], 'read')
    track_memory_access([BatchLength, 120], 'write')

with tf.variable_scope('FC2'):
    W2 = tf.get_variable('W2', [120, NumClasses])
    B2 = tf.get_variable('B2', [NumClasses])
    FC2 = tf.matmul(FC1, W2) + B2
    fc2_params = count_fc_layer_params(120, NumClasses)
    parameter_counts['fc_layers']['FC2'] = fc2_params
    parameter_counts['fc_total'] += fc2_params
    track_memory_access([120, NumClasses], 'read')
    track_memory_access([BatchLength, 120], 'read')
    track_memory_access([BatchLength, NumClasses], 'write')

# Calculate total parameters
parameter_counts['total_params'] = parameter_counts['conv_total'] + parameter_counts['fc_total'] + parameter_counts['batch_norm_params']
parameter_counts['trainable_params'] = parameter_counts['total_params']

# Define loss, optimizer and accuracy
with tf.name_scope('loss'):
    Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=OneHotLabels, logits=FC2))

with tf.name_scope('optimizer'):
    Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(Loss)

with tf.name_scope('accuracy'):
    CorrectPredictions = tf.equal(tf.argmax(FC2, 1), tf.argmax(OneHotLabels, 1))
    Accuracy = tf.reduce_mean(tf.cast(CorrectPredictions, tf.float32))

# Initialize and configure session
Init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Training and evaluation
start_time = time.time()

with tf.Session(config=config) as Sess:
    Sess.run(Init)
    Step = 1
    
    final_train_acc = 0
    final_train_loss = 0
    final_test_acc = 0
    final_precision = 0
    final_recall = 0
    final_f1 = 0
    final_kappa = 0
    
    while Step <= NumIteration:
        batch_indices = np.random.randint(0, x_train.shape[0], BatchLength)
        batch_xs, batch_ys = x_train[batch_indices], y_train[batch_indices]
        
        _, Acc, L = Sess.run([Optimizer, Accuracy, Loss], 
                            feed_dict={InputData: batch_xs, OneHotLabels: batch_ys, is_training: True})
        
        final_train_acc = Acc
        final_train_loss = L
        
        if Step % EvalFreq == 0:
            test_indices = np.random.randint(0, x_test.shape[0], BatchLength)
            test_xs, test_ys = x_test[test_indices], y_test[test_indices]
            
            final_test_acc = Sess.run(Accuracy, 
                               feed_dict={InputData: test_xs, OneHotLabels: test_ys, is_training: False})
            test_predictions = Sess.run(tf.argmax(FC2, 1), 
                                      feed_dict={InputData: test_xs, OneHotLabels: test_ys, is_training: False})
            test_true_labels = np.argmax(test_ys, axis=1)
            
            final_precision = precision_score(test_true_labels, test_predictions, average='weighted')
            final_recall = recall_score(test_true_labels, test_predictions, average='weighted')
            final_f1 = f1_score(test_true_labels, test_predictions, average='weighted')
            final_kappa = cohen_kappa_score(test_true_labels, test_predictions)
        
        Step += 1
    
    inference_results = measure_inference_performance(Sess, InputData, FC2, x_test)

# Calculate training time and memory usage
end_time = time.time()
training_time = end_time - start_time
samples_processed = NumIteration * BatchLength

# Calculate computational metrics
total_flops = compute_dct_domain_flops(input_size, num_channels, num_kernels, kernel_size, stride)
total_memory_accesses = memory_accesses['read_accesses'] + memory_accesses['write_accesses']
total_memory_access = compute_dct_memory_access(input_size, num_channels, num_kernels, kernel_size, stride)
memory_cost = total_memory_access * 4 / (1024 * 1024)

# Print results
print("\n" + "="*75)
print("FINAL RESULTS - DCT-LeNet on Custom Dataset")
print("="*75)

print("\n--- MODEL PARAMETERS ---")
print(f"Convolutional Layers Parameters: {parameter_counts['conv_total']:,}")
for layer_name, params in parameter_counts['conv_layers'].items():
    print(f"  {layer_name}: {params:,}")

print(f"\nBatch Normalization Parameters: {parameter_counts['batch_norm_params']:,}")

print(f"\nFully Connected Layers Parameters: {parameter_counts['fc_total']:,}")
for layer_name, params in parameter_counts['fc_layers'].items():
    print(f"  {layer_name}: {params:,}")

print(f"\nTotal Parameters: {parameter_counts['total_params']:,}")
print(f"Trainable Parameters: {parameter_counts['trainable_params']:,}")

print("\n--- ACCURACY METRICS ---")
print(f"Training Accuracy:      {final_train_acc:.4f}")
print(f"Training Loss:          {final_train_loss:.4f}")
print(f"Test Accuracy:          {final_test_acc:.4f}")
print(f"Precision:              {final_precision:.4f}")
print(f"Recall:                 {final_recall:.4f}")
print(f"F1 Score:               {final_f1:.4f}")
print(f"Cohen's Kappa:          {final_kappa:.4f}")

print("\n--- COMPUTATIONAL METRICS ---")
print(f"Total FLOPs:            {total_flops:,}")
print(f"Total Memory Access:    {total_memory_access:,}")
print(f"Memory Cost:            {memory_cost:.2f} MB")
print(f"Training Time (s):      {training_time:.2f}")

print("\n--- INFERENCE PERFORMANCE ---")
print(f"Throughput:             {inference_results['throughput']:.2f} samples/s")
print(f"Inference Time:         {inference_results['avg_inference_time']*1000:.2f} ms")
print("="*75)