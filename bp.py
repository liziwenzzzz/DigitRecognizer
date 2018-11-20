import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])  # x 不是一个特定的值，而是一个占位符 placeholder ，我们在TensorFlow运行计算时输入这个值

y_ = tf.placeholder(tf.float32, shape=[None, 10])  # 用于输入正确值

W = tf.Variable(tf.zeros([784, 10]))  # 初始化权值W

b = tf.Variable(tf.zeros([10]))  # 初始化偏置项b

# tf.matmul(X，W) 表示 x 乘以 W ，对应之前等式里面的Wx+b;这里 x 是一个2维张量拥有多个输入。然后再加上 b ，把和输入到 tf.nn.softmax 函数里面。
y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))  # 求交叉熵

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  # 用梯度下降法使得交叉熵最小
# argmax 给出某个tensor对象在某一维上的其数据最大值所在的索引值。由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # 检测我们的预测是否真实标签匹配(索引位置一样表示匹配)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 确定正确预测项的比例，我们可以把布尔值转换成浮点数，然后取平均值。
print(accuracy)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):  # 训练阶段，迭代1000次
        batch_xs, batch_ys = mnist.train.next_batch(100)  # 按批次训练，每批100行数据
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  # 执行训练
        if (i % 50 == 0):  # 每训练100次，测试一次
            print("accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))