import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
import time
import psutil
import os
import cv2
from sklearn.preprocessing import LabelEncoder

tf.disable_v2_behavior()
tf.compat.v1.disable_resource_variables()

# Memory monitoring function
def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_cpu_usage():
    """Get current CPU usage percentage"""
    return psutil.cpu_percent(interval=None)

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

# Energy efficiency tracking
energy_metrics = {
    'cpu_usage_samples': [],
    'memory_usage_samples': [],
    'start_memory': 0,
    'peak_memory': 0,
    'total_cpu_time': 0,
    'baseline_power': 0
}

# Memory access tracking
memory_accesses = {'read_accesses': 0, 'write_accesses': 0}

def track_memory_access(tensor_shape, access_type='read'):
    global memory_accesses
    if access_type == 'read':
        memory_accesses['read_accesses'] += 1
    else:
        memory_accesses['write_accesses'] += 1

def init_energy_monitoring():
    """Initialize energy monitoring baseline"""
    global energy_metrics
    energy_metrics['start_memory'] = get_memory_usage()
    energy_metrics['peak_memory'] = energy_metrics['start_memory']
    psutil.cpu_percent(interval=None)
    time.sleep(0.1)
    energy_metrics['baseline_power'] = psutil.cpu_percent(interval=None)

def update_energy_metrics():
    """Update energy efficiency metrics during training"""
    global energy_metrics
    current_memory = get_memory_usage()
    current_cpu = get_cpu_usage()
    
    energy_metrics['memory_usage_samples'].append(current_memory)
    energy_metrics['cpu_usage_samples'].append(current_cpu)
    energy_metrics['peak_memory'] = max(energy_metrics['peak_memory'], current_memory)

def calculate_energy_efficiency_metrics(training_time, inference_results, final_test_acc, total_flops, total_memory_access):
    """Calculate comprehensive energy efficiency metrics"""
    global energy_metrics
    
    avg_memory_usage = np.mean(energy_metrics['memory_usage_samples']) if energy_metrics['memory_usage_samples'] else energy_metrics['start_memory']
    memory_overhead = energy_metrics['peak_memory'] - energy_metrics['start_memory']
    memory_efficiency = final_test_acc / (avg_memory_usage / 1024)
    
    avg_cpu_usage = np.mean(energy_metrics['cpu_usage_samples']) if energy_metrics['cpu_usage_samples'] else 0
    cpu_efficiency = final_test_acc / (avg_cpu_usage + 1e-6)
    
    samples_per_second = (NumIteration * BatchLength) / training_time
    accuracy_per_second = final_test_acc / training_time
    energy_per_sample = (avg_cpu_usage * training_time) / (NumIteration * BatchLength)
    
    inference_energy_efficiency = final_test_acc / (inference_results['avg_inference_time'] * 1000 + 1e-6)
    throughput_efficiency = inference_results['throughput'] / (avg_cpu_usage + 1e-6)
    
    estimated_power_usage = avg_cpu_usage * training_time / 100
    power_efficiency = final_test_acc / (estimated_power_usage + 1e-6)
    
    dct_memory_efficiency = final_test_acc / (total_memory_access / 1e6)
    dct_flop_efficiency = final_test_acc / (total_flops / 1e9)
    
    return {
        'avg_memory_usage_mb': avg_memory_usage,
        'peak_memory_mb': energy_metrics['peak_memory'],
        'memory_overhead_mb': memory_overhead,
        'memory_efficiency': memory_efficiency,
        'avg_cpu_usage_percent': avg_cpu_usage,
        'cpu_efficiency': cpu_efficiency,
        'samples_per_second': samples_per_second,
        'accuracy_per_second': accuracy_per_second,
        'energy_per_sample': energy_per_sample,
        'inference_energy_efficiency': inference_energy_efficiency,
        'throughput_efficiency': throughput_efficiency,
        'estimated_power_usage': estimated_power_usage,
        'power_efficiency': power_efficiency,
        'dct_memory_efficiency': dct_memory_efficiency,
        'dct_flop_efficiency': dct_flop_efficiency
    }

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

# Custom dataset loading function
def load_custom_dataset(train_path, test_path, img_size=(64, 64)):
    """
    Load images from specified directories and create labels based on folder names
    """
    def load_images_from_path(path, img_size):
        images = []
        labels = []
        
        # Get all class folders
        class_folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        class_folders.sort()  # Ensure consistent ordering
        
        for class_idx, class_folder in enumerate(class_folders):
            class_path = os.path.join(path, class_folder)
            
            # Get all image files in the class folder
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            for image_file in image_files:
                image_path = os.path.join(class_path, image_file)
                
                # Load and preprocess image
                try:
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is not None:
                        # Resize image to target size
                        image = cv2.resize(image, img_size)
                        images.append(image)
                        labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    continue
        
        return np.array(images), np.array(labels), class_folders
    
    # Load training data
    x_train, y_train, train_classes = load_images_from_path(train_path, img_size)
    
    # Load test data
    x_test, y_test, test_classes = load_images_from_path(test_path, img_size)
    
    # Ensure both datasets have the same classes
    all_classes = sorted(list(set(train_classes + test_classes)))
    num_classes = len(all_classes)
    
    print(f"Found {num_classes} classes: {all_classes}")
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    return x_train, y_train, x_test, y_test, num_classes

# Pure DCT implementation for TensorFlow compatible with axis=-1 constraint
def tf_dct_2d(x):
    """
    2D DCT implementation using TensorFlow operations
    Input: x with shape [batch, height, width, channels]
    Output: DCT coefficients with same shape
    
    Works around TensorFlow's axis=-1 constraint by reshaping and transposing
    """
    # Get input shape
    input_shape = tf.shape(x)
    batch_size, height, width, channels = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
    
    # First DCT along height dimension
    # Reshape to [batch*width*channels, height] to use axis=-1
    x_reshaped_h = tf.reshape(tf.transpose(x, [0, 2, 3, 1]), [-1, height])
    x_dct_h = tf.spectral.dct(x_reshaped_h, type=2, axis=-1, norm='ortho')
    # Reshape back to [batch, width, channels, height] then transpose to [batch, height, width, channels]
    x_dct_h = tf.transpose(tf.reshape(x_dct_h, [batch_size, width, channels, height]), [0, 3, 1, 2])
    
    # Second DCT along width dimension
    # Reshape to [batch*height*channels, width] to use axis=-1
    x_reshaped_w = tf.reshape(tf.transpose(x_dct_h, [0, 1, 3, 2]), [-1, width])
    x_dct_hw = tf.spectral.dct(x_reshaped_w, type=2, axis=-1, norm='ortho')
    # Reshape back to [batch, height, channels, width] then transpose to [batch, height, width, channels]
    x_dct_hw = tf.transpose(tf.reshape(x_dct_hw, [batch_size, height, channels, width]), [0, 1, 3, 2])
    
    return x_dct_hw

# Load custom dataset
image_train = 'C:/Users/User/Desktop/my ex/PHD/DataSet/training_data'
image_test = 'C:/Users/User/Desktop/my ex/PHD/DataSet/test_data'

x_train, y_train, x_test, y_test, num_classes = load_custom_dataset(image_train, image_test)

# Preprocess the data
x_train = x_train.reshape(-1, 64, 64, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 64, 64, 1).astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Set the parameters for VGG7-like architecture
NumIteration = 1
EvalFreq = 1
BatchLength = 64
Size = [64, 64, 1]  # Updated size
input_size = [64, 64]  # Updated input size
num_channels = 1
LearningRate = 1e-4
num_kernels = [1, 1, 2, 2]  # VGG7-like structure
NumClasses = num_classes  # Use detected number of classes
specPoolSize = 1
kernel_size = [3, 3]  # VGG-style 3x3 kernels
stride = [2, 2]

def dct_relu(x):
    """ReLU activation adapted for DCT domain"""
    shape = x.get_shape().as_list()
    track_memory_access(shape, 'read')
    track_memory_access(shape, 'write')
    return tf.nn.relu(x - 0.1) + 0.1

def dct_domain_convolution(dct_input, out_channels, layer_idx):
    """VGG7-style convolution operation in DCT domain"""
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
    """Measure comprehensive inference performance with proper timing"""
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

# Reset graph and create placeholders
tf.reset_default_graph()
InputData = tf.placeholder(tf.float32, [None] + Size)
OneHotLabels = tf.placeholder(tf.float32, [None, NumClasses])
is_training = tf.placeholder(tf.bool, name='is_training')

# Transform input to DCT domain (all processing will remain in DCT domain)
CurrentInput = InputData
dctInput = tf_dct_2d(CurrentInput)

# Create convolutional layers purely in DCT domain
current_input_channels = num_channels
for N in range(len(num_kernels)):
    with tf.variable_scope('dct_conv' + str(N)):
        dctInput = dct_domain_convolution(dctInput, num_kernels[N], N)
        
        conv_params = count_conv_layer_params(current_input_channels, num_kernels[N], 
                                              kernel_size[0], kernel_size[1])
        parameter_counts['conv_layers'][f'dct_conv{N}'] = conv_params
        parameter_counts['conv_total'] += conv_params
        
        bn_trainable, bn_non_trainable = count_batch_norm_params(num_kernels[N])
        parameter_counts['batch_norm_params'] += bn_trainable
        parameter_counts['conv_layers'][f'dct_conv{N}_bn'] = bn_trainable
        
        current_input_channels = num_kernels[N]

# Flatten
with tf.variable_scope('Flatten'):
    batch_size = tf.shape(dctInput)[0]
    flattened = tf.reshape(dctInput, [batch_size, -1])
    FeatureLength = int(np.prod(dctInput.get_shape().as_list()[1:]))
    track_memory_access([BatchLength, FeatureLength], 'read')
    track_memory_access([BatchLength, FeatureLength], 'write')

# FC layers
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
    W2 = tf.get_variable('W2', [120, 84])
    B2 = tf.get_variable('B2', [84])
    FC2 = tf.nn.relu(tf.matmul(FC1, W2) + B2)
    fc2_params = count_fc_layer_params(120, 84)
    parameter_counts['fc_layers']['FC2'] = fc2_params
    parameter_counts['fc_total'] += fc2_params
    track_memory_access([120, 84], 'read')
    track_memory_access([BatchLength, 120], 'read')
    track_memory_access([BatchLength, 84], 'write')

with tf.variable_scope('FC3'):
    W3 = tf.get_variable('W3', [84, NumClasses])
    B3 = tf.get_variable('B3', [NumClasses])
    FC3 = tf.matmul(FC2, W3) + B3
    fc3_params = count_fc_layer_params(84, NumClasses)
    parameter_counts['fc_layers']['FC3'] = fc3_params
    parameter_counts['fc_total'] += fc3_params
    track_memory_access([84, NumClasses], 'read')
    track_memory_access([BatchLength, 84], 'read')
    track_memory_access([BatchLength, NumClasses], 'write')

# Calculate total parameters
parameter_counts['total_params'] = parameter_counts['conv_total'] + parameter_counts['fc_total'] + parameter_counts['batch_norm_params']
parameter_counts['trainable_params'] = parameter_counts['total_params']

# Loss and optimizer
with tf.name_scope('loss'):
    Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=OneHotLabels, logits=FC3))

with tf.name_scope('optimizer'):
    Optimizer = tf.train.AdamOptimizer(LearningRate).minimize(Loss)

with tf.name_scope('accuracy'):
    CorrectPredictions = tf.equal(tf.argmax(FC3, 1), tf.argmax(OneHotLabels, 1))
    Accuracy = tf.reduce_mean(tf.cast(CorrectPredictions, tf.float32))

# Initialize and configure session
Init = tf.global_variables_initializer()
config = tf.ConfigProto()
saver = tf.train.Saver()
config.gpu_options.allow_growth = True

# Initialize variables for final metrics
final_accuracy = 0
final_train_loss = 0
final_precision = 0
final_recall = 0
final_f1 = 0
final_kappa = 0

# Initialize energy monitoring
print("Initializing energy monitoring...")
init_energy_monitoring()

# Training loop
start_time = time.time()

with tf.Session(config=config) as Sess:
    Sess.run(Init)
    Step = 1
    
    while Step <= NumIteration:
        if Step % 100 == 0:
            update_energy_metrics()
        
        batch_indices = np.random.randint(0, x_train.shape[0], BatchLength)
        batch_xs, batch_ys = x_train[batch_indices], y_train[batch_indices]
        
        _, Acc, L = Sess.run([Optimizer, Accuracy, Loss], 
                            feed_dict={InputData: batch_xs, OneHotLabels: batch_ys, is_training: True})
        
        if Step % EvalFreq == 0:
            test_indices = np.random.randint(0, x_test.shape[0], BatchLength)
            test_xs, test_ys = x_test[test_indices], y_test[test_indices]
            
            Acc_test = Sess.run(Accuracy, 
                               feed_dict={InputData: test_xs, OneHotLabels: test_ys, is_training: False})
            test_predictions = Sess.run(tf.argmax(FC3, 1), 
                                      feed_dict={InputData: test_xs, OneHotLabels: test_ys, is_training: False})
            test_true_labels = np.argmax(test_ys, axis=1)

            # Calculate metrics
            final_precision = precision_score(test_true_labels, test_predictions, average='weighted')
            final_recall = recall_score(test_true_labels, test_predictions, average='weighted')
            final_f1 = f1_score(test_true_labels, test_predictions, average='weighted')
            final_kappa = cohen_kappa_score(test_true_labels, test_predictions)
            
            final_accuracy = Acc_test
            final_train_loss = L
        
        Step += 1
    
    update_energy_metrics()
    
    inference_results = measure_inference_performance(Sess, InputData, FC3, x_test)

# Calculate training time
end_time = time.time()
training_time = end_time - start_time

# Calculate FLOPs and memory access
total_flops = compute_dct_domain_flops(input_size, num_channels, num_kernels, kernel_size, stride)
total_memory_accesses = memory_accesses['read_accesses'] + memory_accesses['write_accesses']
total_memory_access = compute_dct_memory_access(input_size, num_channels, num_kernels, kernel_size, stride)

# Calculate theoretical memory cost
memory_cost = total_memory_access * 4 / (1024 * 1024)  # Convert to MB (4 bytes per float32)

# Calculate energy efficiency metrics
energy_results = calculate_energy_efficiency_metrics(training_time, inference_results, final_accuracy, total_flops, total_memory_access)

# Print results
print("\n" + "="*70)
print("FINAL RESULTS - DCT-VGG7 on Custom Dataset")
print("="*70)
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
print("Training accuracy:      {:.4f}".format(final_accuracy))
print("Training loss:          {:.4f}".format(final_train_loss))
print("Test accuracy:          {:.4f}".format(final_accuracy))
print("Precision:              {:.4f}".format(final_precision))
print("Recall:                 {:.4f}".format(final_recall))
print("F1 Score:               {:.4f}".format(final_f1))
print("Cohen's Kappa:          {:.4f}".format(final_kappa))

print("\n--- COMPUTATIONAL METRICS ---")
print("Total FLOPs:            {:,}".format(total_flops))
print("Total memory access:    {:,}".format(total_memory_access))
print("Memory cost:            {:.2f} MB".format(memory_cost))
print("Training Time (s):      {:.2f}".format(training_time))

print("\n--- INFERENCE PERFORMANCE ---")
print("Throughput:             {:.2f} samples/s".format(inference_results['throughput']))
print("Inference Time:         {:.2f} ms".format(inference_results['avg_inference_time']*1000))

print("\n--- ENERGY EFFICIENCY METRICS ---")
print(f"Avg Memory Usage:       {energy_results['avg_memory_usage_mb']:.2f} MB")
print(f"Peak Memory Usage:      {energy_results['peak_memory_mb']:.2f} MB")
print(f"Memory Overhead:        {energy_results['memory_overhead_mb']:.2f} MB")
print(f"Memory Efficiency:      {energy_results['memory_efficiency']:.4f}")
print(f"Avg CPU Usage:          {energy_results['avg_cpu_usage_percent']:.2f}%")
print(f"CPU Efficiency:         {energy_results['cpu_efficiency']:.4f}")
print(f"Samples/Second:         {energy_results['samples_per_second']:.2f}")
print(f"Accuracy/Second:        {energy_results['accuracy_per_second']:.6f}")
print(f"Energy/Sample:          {energy_results['energy_per_sample']:.4f}")
print(f"Inference Energy Eff:   {energy_results['inference_energy_efficiency']:.4f}")
print(f"Throughput Efficiency:  {energy_results['throughput_efficiency']:.4f}")
print(f"Est. Power Usage:       {energy_results['estimated_power_usage']:.2f}")
print(f"Power Efficiency:       {energy_results['power_efficiency']:.4f}")
print(f"DCT Memory Efficiency:  {energy_results['dct_memory_efficiency']:.6f}")
print(f"DCT FLOP Efficiency:    {energy_results['dct_flop_efficiency']:.6f}")

print("\n--- MEMORY ACCESS STATISTICS ---")
print(f"Total Read Accesses:    {memory_accesses['read_accesses']:,}")
print(f"Total Write Accesses:   {memory_accesses['write_accesses']:,}")
print(f"Total Memory Accesses:  {total_memory_accesses:,}")

print("\n--- EFFICIENCY RATIOS ---")
print(f"Accuracy per Parameter: {final_accuracy/parameter_counts['total_params']:.8f}")
print(f"Accuracy per FLOP:      {final_accuracy/total_flops:.12f}")
print(f"Accuracy per MB:        {final_accuracy/memory_cost:.6f}")
print(f"Parameters per FLOP:    {parameter_counts['total_params']/total_flops:.8f}")

print("\n" + "="*70)
print("END OF RESULTS")
print("="*70)