/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator.h"
#include "../utility/visualization.h"

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins");
    initThreadFlag = false;
    clearState();
}

Estimator::~Estimator()
{
    if (MULTIPLE_THREAD)
    {
        processThread.join();
        printf("join thread \n");
    }
}

void Estimator::clearState()
{
    mProcess.lock();
    while(!accBuf.empty())
        accBuf.pop();
    while(!gyrBuf.empty())
        gyrBuf.pop();
    while(!featureBuf.empty())
        featureBuf.pop();

    prevTime = -1;
    curTime = 0;
    openExEstimation = 0;
    initP = Eigen::Vector3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;

    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
        {
            delete pre_integrations[i];
        }
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();

    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;

    mProcess.unlock();
}

void Estimator::setParameter()
{
    mProcess.lock();
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
        cout << " exitrinsic cam " << i << endl  << ric[i] << endl << tic[i].transpose() << endl;
    }
    f_manager.setRic(ric);
    // 初始化重投影误差的平方根信息矩阵， 它在 非线性优化（Bundle Adjustment, BA） 中起到了“加权”的作用。
    // 投影误差越大 → 权重越高 → 优化器越重视该约束， 乘以焦距 f 是为了把“像素误差”转换为“归一化坐标误差”
    // 这里使用 FOCAL_LENGTH / 1.5 而不是 FOCAL_LENGTH 纯粹是一个经验系数，用来：
    // 1） 避免权重太大导致优化收敛震荡； 2） 在不同相机焦距下保持数值稳定性。
    ProjectionTwoFrameOneCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity(); // 同一相机的两帧之间的观测约束
    ProjectionTwoFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity(); // 不同相机（如双目）之间的约束
    ProjectionOneFrameTwoCamFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity(); // 同一时刻两个相机的约束
    td = TD;
    g = G;
    cout << "set g " << g.transpose() << endl;
    featureTracker.readIntrinsicParameter(CAM_NAMES); // 读取相机内参，生成相应的相机模型

    std::cout << "MULTIPLE_THREAD is " << MULTIPLE_THREAD << '\n';
    if (MULTIPLE_THREAD && !initThreadFlag)
    {
        initThreadFlag = true;
        processThread = std::thread(&Estimator::processMeasurements, this);
    }
    mProcess.unlock();
}

void Estimator::changeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    mProcess.lock();
    if(!use_imu && !use_stereo)
        printf("at least use two sensors! \n");
    else
    {
        if(USE_IMU != use_imu)
        {
            USE_IMU = use_imu;
            if(USE_IMU)
            {
                // reuse imu; restart system
                restart = true;
            }
            else
            {
                if (last_marginalization_info != nullptr)
                    delete last_marginalization_info;

                tmp_pre_integration = nullptr;
                last_marginalization_info = nullptr;
                last_marginalization_parameter_blocks.clear();
            }
        }
        
        STEREO = use_stereo;
        printf("use imu %d use stereo %d\n", USE_IMU, STEREO);
    }
    mProcess.unlock();
    if(restart)
    {
        clearState();
        setParameter();
    }
}

/**
 * @brief 
 * 
 * @param t 当前时间戳
 * @param _img 左目图像
 * @param _img1 右目图像
 */
void Estimator::inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1)
{
    inputImageCnt++;
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    TicToc featureTrackerTime;

    if(_img1.empty())
        featureFrame = featureTracker.trackImage(t, _img);          // 单目跟踪
    else
        featureFrame = featureTracker.trackImage(t, _img, _img1);   // 双目跟踪
    //printf("featureTracker time: %f\n", featureTrackerTime.toc());

    if (SHOW_TRACK) // 发布跟踪图像数据
    {
        cv::Mat imgTrack = featureTracker.getTrackImage();
        pubTrackImage(imgTrack, t);
    }
    
    if(MULTIPLE_THREAD)  // 如果单独开启了处理线程，就直接将跟踪结果放入特征队列中即可
    {     
        if(inputImageCnt % 2 == 0)
        {
            mBuf.lock();
            featureBuf.push(make_pair(t, featureFrame));
            mBuf.unlock();
        }
    }
    else // 否则，手动调用函数进行处理
    {
        mBuf.lock();
        featureBuf.push(make_pair(t, featureFrame));
        mBuf.unlock();
        TicToc processTime;
        processMeasurements();
        printf("process time: %f\n", processTime.toc());
    }
    
}

void Estimator::inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    mBuf.lock();
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    //printf("input imu with time %f \n", t);
    mBuf.unlock();

    // 初始化完成后，再发布相关的里程数据
    if (solver_flag == NON_LINEAR)
    {
        mPropagate.lock();
        fastPredictIMU(t, linearAcceleration, angularVelocity);
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
        mPropagate.unlock();
    }
}

void Estimator::inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame)
{
    mBuf.lock();
    featureBuf.push(make_pair(t, featureFrame));
    mBuf.unlock();

    // 如果没开启处理线程，就手动进行处理
    if(!MULTIPLE_THREAD)
        processMeasurements();
}

// 获取t0到t1的imu数据 
// ...  t<=t0(全部丢弃)  |  (t0, t1) 这些全部输出并pop  |  第一个 >= t1 输出但不pop  |  ...
bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                vector<pair<double, Eigen::Vector3d>> &gyrVector)
{
    if(accBuf.empty())
    {
        printf("not receive imu\n");
        return false;
    }
    //printf("get imu from %f %f\n", t0, t1);
    //printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);
    // 确保接收到的最新的imu时间比image时间新
    if(t1 <= accBuf.back().first)
    {
        // 丢弃早于/等于 t0 的 IMU（对齐区间起点）
        // 注意这里是 <= ，因为 t0 对应的 IMU 数据已经被上一帧图像插值处理过了
        while (accBuf.front().first <= t0)
        {
            accBuf.pop();
            gyrBuf.pop();
        }
        // 收集 (t0, t1) 开区间内的所有 IMU 测量值
        while (accBuf.front().first < t1)
        {
            accVector.push_back(accBuf.front());
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            gyrBuf.pop();
        }
        // note: 让输出数据序列至少包含一个跨过 t1 的样本，以便预积分实现做末端对齐(插值)
        accVector.push_back(accBuf.front());
        gyrVector.push_back(gyrBuf.front());
    }
    else // 最新的imu时间比image的时间还老
    {
        printf("wait for imu\n");
        return false;
    }
    return true;
}

bool Estimator::IMUAvailable(double t)
{
    if(!accBuf.empty() && t <= accBuf.back().first)
        return true;
    else
        return false;
}

void Estimator::processMeasurements()
{
    while (1)
    {
        //printf("process measurments\n");
        pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > feature;
        vector<pair<double, Eigen::Vector3d>> accVector, gyrVector;
        // featureBuf是pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > >
        // 第一个double是图像时间戳，第二个map是特征点数据(应该叫TimedFeature更合适些)
        // 即 map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
        // 特征点id - 左右双目特征数据的 关联容器
        // 特征数据格式: (相机id, [x,y,z,p_u,p_v,velocity_x,velocity_y])
        if(!featureBuf.empty())
        {
            feature = featureBuf.front();
            curTime = feature.first + td;
            // 等待直到IMU数据足够
            // VIO 的状态递推依赖IMU。要处理时刻tk的图像，必须拥有直到tk时刻为止的所有IMU数据
            // 如果图像来了，但IMU数据还没传过来（或者IMU频率不够快、网络延迟），这里会死循环等待，直到IMU数据“追上”图像时间
            // 确保后续的预积分（Pre-integration）能完整覆盖两帧图像之间的时间段
            while(1)
            {
                // 如果不用IMU，或者IMU数据已经涵盖了当前图像时间戳，就可以退出等待
                if ((!USE_IMU  || IMUAvailable(feature.first + td)))
                    break;
                else
                {
                    printf("wait for imu ... \n");
                    if (! MULTIPLE_THREAD)
                        return;
                    std::chrono::milliseconds dura(5);
                    std::this_thread::sleep_for(dura);
                }
            }
            // 数据准备: 取出IMU区间数据，消费一帧图像特征
            mBuf.lock();
            if(USE_IMU)
                // 取出两帧图像之间的所有IMU数据, 如果是第一帧图像，则取出该图像时间戳之前的所有IMU数据
                getIMUInterval(prevTime, curTime, accVector, gyrVector);

            featureBuf.pop();
            mBuf.unlock();

            if(USE_IMU)
            {
                if(!initFirstPoseFlag) // IMU静止初始化(需要先检查是否为静止)
                    initFirstIMUPose(accVector);
                // 预积分两帧图像之间的所有IMU数据，并推进状态到当前图像时刻curTime(而不是IMU最后一个样本时刻)
                // 如果是第一帧图像，则从第一帧IMU时刻推进到第一帧图像时刻curTime
                for(size_t i = 0; i < accVector.size(); i++)
                {
                    // 这里的积分步长 dt 的构造很关键，它把 [prevTime, curTime] 切成三类段：
                    // 第一段：prevTime -> 第一个 IMU 样本时刻
                    // 中间段：相邻 IMU 样本之间
                    // 末段：最后一个 IMU 样本时刻 -> curTime
                    double dt;
                    if(i == 0)
                        dt = accVector[i].first - prevTime;
                    else if (i == accVector.size() - 1)
                        dt = curTime - accVector[i - 1].first;
                    else
                        dt = accVector[i].first - accVector[i - 1].first;
                    processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second); // imu预积分
                    // 这里的“最后一次processIMU”常见语义是:
                    // 用“最后一条IMU测量值”作为区间末端附近的代表（近似零阶保持），把状态外推/积分到严格的curTime
                    // 注意它用的是 curTime - accVector[last-1].time，而不是用 accVector[last].time，这等价于“只积分到 curTime，不积分到那条样本的真实时间（可能略晚于 curTime）”
                    // 这也是为什么 getIMUInterval() 要确保有一个 >= curTime 的样本：没有它，末端外推会缺少测量
                }
            }
            mProcess.lock();
            processImage(feature.second, feature.first);
            prevTime = curTime;

            printStatistics(*this, 0);

            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(feature.first);

            pubOdometry(*this, header);
            pubKeyPoses(*this, header);
            pubCameraPose(*this, header);
            pubPointCloud(*this, header);
            pubKeyframe(*this);
            pubTF(*this, header);
            mProcess.unlock();
        }

        if (! MULTIPLE_THREAD)
            break;

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
}

// IMU静止初始化(需要先检查是否为静止)
void Estimator::initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    //return;
    Eigen::Vector3d averAcc(0, 0, 0);
    int n = (int)accVector.size();
    for(size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n; // 均值(需要保持静止)
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    Matrix3d R0 = Utility::g2R(averAcc);
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0; // 静止初始化的yaw设置为0
    Rs[0] = R0; // 只初始化旋转
    cout << "init R0 " << endl << Rs[0] << endl;
    //Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}

/**
 * @brief 处理每一帧原始IMU数据
 * 1. 对IMU数据进行预积分(Pre-integration)，得到两帧图像之间IMU的增量测量，作为后续非线性优化的约束项（Factor）
 * 2. 利用IMU数据对状态进行递推(Propagation), 利用中值积分法，实时更新当前时刻系统在世界坐标系下的位置、速度和姿态（P, V, Q），作为优化的初始猜测值（Initial Guess）
 * 第 0 帧没有“前一帧”，所以第 0 帧不形成“帧间 IMU 因子”
 * frame_count == 0：只更新 acc_0/gyr_0，不累计预积分、不传播 Ps/Vs/Rs
 * frame_count != 0：开始累积从上一帧到当前帧的预积分，并传播状态
 * @param t 当前时刻
 * @param dt 与上一帧IMU数据的时间间隔
 * @param linear_acceleration 加速度数据
 * @param angular_velocity 角速度数据
 */
void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu) // 第一帧IMU数据
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        // 每次新建预积分对象时，这里的acc_0、gyr_0就是在上一图像帧时间戳img_t处对应的imu观测量
        // 不过，这里的acc_0、gyr_0并不是上一图像帧时刻的真实imu观测量，也不是“离散积分”过程中通过线性插值得到的近似值，而是在img_t时刻后的第一个imu观测量(>=img_t)
        // 这里构造的pre_integrations[frame_count]和构造tmp_pre_integration的是一样的参数值
        // pre_integrations[frame_count]: 第 frame_count 帧对应的预积分器（通常表示从第 frame_count-1 帧到第 frame_count 帧的 IMU 约束在累积）
        // 当frame_count为0时，用第一个IMU数据初始化pre_integrations[0]，但不进行预积分和状态传播
        // 当frame_count大于0时，pre_integrations[frame_count]就用上一帧图像对应的img_t的IMU数据进行初始化
        // tmp_pre_integration：临时预积分（常用于失败重启/重求解/非线性迭代中的快速尝试）
        // Bas[j], Bgs[j]：第 j 帧的加速度计/陀螺偏置估计（滑窗状态的一部分）
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    // 第一帧图像数据不需要进行预积分，因为预积分是用来进行相邻两图像帧之间的约束
    if (frame_count != 0)
    {
        // 计算IMU预积分
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity); // push_back是重载函数
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);
        // 原始 IMU 缓存（用于重传播/边缘化）
        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        // 执行状态递推 (State Propagation)
        // 采用中值法离散积分更新当前图像帧对应的IMU的PVQ，并将积分出来的PVQ作为第j帧图像的状态初始值
        // 噪声是零均值的，所以这里在进行均值传递时就忽略了
        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;   // 上一时刻的加速度信息（减去了重力向量），基于世界坐标系
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j]; // 中值角速度 = (gyr_old + gyr_new) / 2 - bias 这个是基于IMU坐标系的
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g; // 当前时刻的加速度信息（减去了重力向量），基于世界坐标系
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);  // 中值加速度(基于世界坐标系)
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
        // Rs[j]、Ps[j]、Vs[j]保存的是从IMU body系到惯性世界坐标系的PVQ，且在这里计算时，认为偏置bias是固定为初始值的
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity; 
    // 为什么既做“预积分”又做“状态传播”???
    // 预积分（IntegrationBase）：是给滑窗优化用的 IMU 因子（约束相邻关键帧状态），优化时会根据偏置线性化点反复修正
    // 状态传播（更新 Rs/Ps/Vs）：是在线实时给当前帧一个合理初值，否则视觉更新/非线性优化很难收敛或需要更多迭代
}

/**
 * @brief 
 * image是当前帧(包括左右双目)所有图像特征点id与左右双目特征点对应信息的键值对，(feature_id, [camera_id, [x,y,z,u,v,vx,vy]])
 * 其中特征点有以下信息: 
 * 1. 所有(左右双目)特征点去畸变后 在相机归一化平面的坐标(x,y,1);
 * 2. 所有(左右双目)特征点去畸变前 在图像平面的像素坐标(u,v);
 * 3. 所有(左右双目)特征点去畸变后 在相机归一化平面的速度(vx, vy).
 * @param image 提取出的图像特征数据
 * @param header 
 */
void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size()); // image.size() 是当前帧图像中跟踪到的特征点数量
    // 为了维持滑动窗口的大小，需要去除旧的帧而添加新的帧，也就是边缘化；但是注意，边缘化一定是在滑动窗口填满之后才进行的
    // 判断次新帧是否作为关键帧，以此来确定滑窗的时候是边缘化掉滑动窗口中的最老帧还是次新帧
    if (f_manager.addFeatureCheckParallax(frame_count, image, td)) // 添加特征点到feature中，并根据路标点跟踪的次数和视差，判断次新帧是否是关键帧
    {
        marginalization_flag = MARGIN_OLD; // 次新帧是关键帧，则边缘化掉最老帧
        //printf("keyframe\n");
    }
    else
    {
        marginalization_flag = MARGIN_SECOND_NEW; // 次新帧不是关键帧，则直接丢掉次新帧
        //printf("non-keyframe\n");
    }

    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount());
    Headers[frame_count] = header; // Headers数组只保存滑动窗口中的图像帧的时间，在后面会丢掉滑动窗口之外的数据

    ImageFrame imageframe(image, header);
    imageframe.pre_integration = tmp_pre_integration; // 第一帧图像的tmp_pre_integration为NULL
    all_image_frame.insert(make_pair(header, imageframe)); // all_image_frame 中元素的删除在 SlideWindow()
    // 这里的acc_0、gyr_0是在当前图像帧时间戳img_t处对应的imu观测量
    // 其实并不是严格对应，因为IMU数据是离散的，而图像时间戳通常落在两个IMU样本之间 参考取imu区间的函数 getIMUInterval() 会大于时间戳
    // 是否考虑进行插值???
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    if(ESTIMATE_EXTRINSIC == 2) // 没有提供外参， 进行外参在线标定
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0) // 外参在线标定要从第二帧才会开始，因为从第二帧开始才会有预积分数据
        {
            // 获取前一帧与当前帧对应匹配的所有路标点，分别在这两帧下的相机归一化平面坐标
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            // 这里只估计从camera坐标系到IMU坐标系的旋转外参calib_ric，这里没有估计平移外参
            // 相机与IMU之间的旋转外参标定非常重要，偏差1-2°系统的精度就会变的极低
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric))
            {
                // 一般是刚好在进行初始化开始之前，会完成旋转外参的标定
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1; // 先估计出旋转外参后，以这个旋转外参为初始参考值，后续再继续估计
            }
        }
    }
// 简单介绍下滑窗:
// 滑动窗口内的帧索引都是 0 ~ WINDOW_SIZE
// 初始化阶段：系统刚启动，窗口是空的。每来一帧，frame_count 加 1。Ps[], Rs[] 数组慢慢被填满，直到 frame_count == WINDOW_SIZE 时，窗口填满，开始初始化
//           我们在填满窗口的过程中不关心每一帧是否为关键帧，只会在窗口满了再用次新帧是否为关键帧进行滑窗操作
// 稳定运行阶段：初始化完成后，frame_count 固定为 WINDOW_SIZE，窗口满了。每来一帧，滑动窗口就前进一格，frame_count 仍然是 WINDOW_SIZE
//    - 新的一帧进来: 此时实际上有 WINDOW_SIZE + 1 帧数据了
//    - 优化: 进行非线性优化
//    - 滑动窗口: 为了保持计算量恒定，系统必须踢出一帧（要么是最老的一帧，要么是次新帧）
//    - 结果：踢出一帧后，剩下的帧数依然是 WINDOW_SIZE   
    if (solver_flag == INITIAL) // 系统最开始需要进行视觉-惯性联合初始化
    {
        // monocular + IMU initilization 单目+IMU初始化
        // 单目没有尺度信息，必须靠IMU恢复尺度
        if (!STEREO && USE_IMU)
        {
            // 单目初始化需要等滑动窗口填满后才进行，确保有足够多的图像帧参与初始化
            // 窗口满了就开始进行初始化，此时对里面的每一帧是不是关键帧没有要求
            if (frame_count == WINDOW_SIZE)
            {
                bool result = false;
                // 1. 要保证已经成功完成了外参标定，如果是在线估计外参则要先成功估计好外参之后才会进入到初始化中
                // 2. 且当前帧距上一次初始化帧的时间间隔要大于0.1秒，限制初始化尝试频率（避免每帧都跑一次初始化，节省算力并减少抖动）
                // 因为初始化过程不是每次都能成功的，所以这里用initial_timestamp记录上一次尝试初始化的时间戳
                if(ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1)
                {
                    // 进行初始化的目的主要有两个：
                    // 1. 如果没有恢复出一个良好的尺度，就无法对单目相机、IMU这两个传感器做进一步的融合；这主要体现在平移以及三角化得到的路标点的深度；
                    // 因为之后在非线性优化阶段的量都是以IMU积分出来的值作为初值，它天然的带有尺度。
                    // 2. IMU会受到bias的影响，所以要得到IMU的初始bias。
                    // 初始化只进行一次就够了，因为初始化的时候，就能确定尺度scale和bias的初始值，
                    // 尺度scale确定后，在初始化时获得的这些路标点都是准的了，后续通过PnP或者BA得到的特征点都是真实尺度的了。
                    // 而bias初始值确定之后，在后续的非线性优化过程中，会实时更新。
                    result = initialStructure();
                    initial_timestamp = header; // 记录本次初始化尝试的时间戳  
                }
                // 初始化成功
                if(result)
                {
                    optimization();
                    updateLatestStates();
                    solver_flag = NON_LINEAR;
                    slideWindow();
                    ROS_INFO("Initialization finish!");
                }
                else // 如果此次初始化失败，则要直接滑动窗口丢掉一帧，然后再进行初始化，直到初始化完成
                    slideWindow();
            }
        }

        // stereo + IMU initilization 双目+IMU初始化
        if(STEREO && USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            if (frame_count == WINDOW_SIZE)
            {
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++)
                {
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                solveGyroscopeBias(all_image_frame, Bgs);
                for (int i = 0; i <= WINDOW_SIZE; i++)
                {
                    pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
                }
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }

        // stereo only initilization 双目初始化
        if(STEREO && !USE_IMU)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            optimization();

            if(frame_count == WINDOW_SIZE)
            {
                optimization();
                updateLatestStates();
                solver_flag = NON_LINEAR;
                slideWindow();
                ROS_INFO("Initialization finish!");
            }
        }
        // 在初始化完成前，如果窗口没满，系统只是简单地把数据塞进数组，并把上一帧的位姿复制过来占位，等待下一帧数据
        // 从第二帧开始，这里的数据就不是零了，因为在processIMU()中已经对Ps/Vs/Rs做了状态传播
        if(frame_count < WINDOW_SIZE)
        {
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }

    }
    else
    {
        TicToc t_solve;
        if(!USE_IMU)
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
        f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
        optimization();
        set<int> removeIndex;
        outliersRejection(removeIndex);
        f_manager.removeOutlier(removeIndex);
        if (! MULTIPLE_THREAD)
        {
            featureTracker.removeOutliers(removeIndex);
            predictPtsInNextFrame();
        }
            
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection())
        {
            ROS_WARN("failure detection!");
            failure_occur = 1;
            clearState();
            setParameter();
            ROS_WARN("system reboot!");
            return;
        }

        slideWindow();
        f_manager.removeFailures();
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates();
    }  
}

/**
 * @brief  VINS初始化采用视觉和IMU的松耦合初始化方案，主要分两步：
 * 1. 在未知尺度的情况下，首先通过纯视觉SFM求解滑窗内所有帧的位姿，以及所有路标点的3D位置。
 * 2. 和IMU预积分进行对齐，求解重力、尺度、陀螺仪bias、以及每一帧的速度。
 * 在不知道尺度的情况下，先建立一个纯视觉的 3D 结构（SfM），然后将这个视觉结构与 IMU 的预积分结果进行“对齐”，
 * 从而恢复出物理尺度、重力方向、速度以及陀螺仪偏置。
 * @return true 
 * @return false 
 */
bool Estimator::initialStructure()
{
    TicToc t_sfm;
    // 通过线加速度的标准差来判断IMU是否有充分的运动激励。
    // 对于单目系统而言，视觉信息只能获得二维信息，损失了深度信息，所以需要相机动一下，通过三角化才能获得损失的深度信息，
    // 但是三角化恢复的这个深度信息，它的尺度是随机的，不是真实物理的，所以还需要IMU来标定这个尺度，
    // 而要想让IMU标定这个尺度，IMU也需要平移运动一下，得到PVQ中的P。
    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1); // 计算所有帧的平均加速度
        // 计算加速度的方差
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1)); // 标准差
        //ROS_WARN("IMU variation %f!", var);
        // 如果设备静止，加速度计测量的主要是重力，方差应该非常小（只有噪声）。
        // 如果设备有运动（旋转或平移），加速度会有剧烈波动，方差变大
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1]; // 记录初始化时滑窗中每一帧的姿态(相对于起始帧)
    Vector3d T[frame_count + 1];    // 记录初始化时滑窗中每一帧的位置(相对于起始帧)
    map<int, Vector3d> sfm_tracked_points; // 记录初始化时所有成功三角化的路标点(相对于起始帧), key: feature_id, value: 3d point
    vector<SFMFeature> sfm_f; // 存储滑窗中所有特征点观测信息的容器
    // 遍历滑窗中的所有特征点，将特征管理器(f_manager)中的数据转换为SfM需要的格式
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    } 
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    // 从滑窗中找到距离当前帧具有足够视差的参考帧l，并由本质矩阵恢复出R、t作为初始值
    // l帧指的是，在滑动窗口中，从第一帧开始，首个满足与当前帧的平均视差足够大的帧，并且它还会做为参考帧在下面的全局sfm中使用。（找出的这个l一般都等于0）
    // 此处的relative_R，relative_T是当前帧到参考帧cl（l帧）的坐标变换R、t
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    // 求解纯视觉SFM问题，这里将l帧作为参考帧，得到滑动窗口中的每一图像帧到l帧（参考帧）的姿态四元数Q、平移向量T和成功三角化的3d路标点坐标sfm_tracked_points
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l, // 这里为什么是frame_count + 1? 因为这里传入的是当前滑窗中的图片总帧数
              relative_R, relative_T,
              sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD; // 若求解SFM问题失败，则边缘化最老帧并滑动窗口
        return false;
    }

    // 注意在前面针对的都是在当前滑动窗口中的帧
    // 在之前的步骤中，GlobalSFM 仅解算了滑动窗口中关键帧(或被当作关键帧的帧)的位姿（即 Headers 数组里记录的那几帧）
    // 但是，VINS 的缓存队列 all_image_frame 中保存了过去几秒内的所有图像帧（可能包含很多非关键帧/中间帧）。
    // 为了后续能精确地跟 IMU 积分（它是连续的）进行对比，我们需要知道每一帧的位姿，而不仅仅是关键帧的位姿。
    // 因此，这段代码的任务是：利用 SfM 已经建立好的 3D 地图点，通过 PnP 算法，把那些没参与 SfM 的“中间帧”的位姿也都算出来
    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++) // 遍历所有的图像帧
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        // 情况 A：当前帧是 SfM 算过的“关键帧”
        if((frame_it->first) == Headers[i]) // 该帧在滑动窗口中，直接使用前面SFM结果
        {
            // SfM 算的是相机在世界下的位姿，而VINS需要的是 IMU 在世界下的位姿
            frame_it->second.is_key_frame = true; // RIC[0] 左目 R_imu_cam
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose(); // R^cl_ci * (R_imu_cam)^T 在前面的sfm中求解的是相对于参考帧cl的旋转，这里要转换到IMU body系下
            frame_it->second.T = T[i]; // 此时尺度s还未估计出来，因此还不能对平移做变换
            i++;
            continue;
        }
        // 对于不在当前滑动窗口中的帧，根据上一步SFM得到的3d路标点，通过PNP求解出该图像帧的位姿
        // 因为在之前的初始化流程中有可能出现失败了，并且当之前的窗口数据满了时，还会进行滑窗丢弃，
        // 丢弃最老帧也可能丢弃次新帧，这就造成了在当前滑动窗口中的帧可能不是全局连续的

        // 情况 B：当前帧是没算过的“中间帧”    如果时间戳不匹配，说明这是一帧中间帧，需要我们自己算
        // 假设 all_image_frame 的时间戳递增，Headers[] 也递增。遇到非关键帧时，i 指向“它附近”的关键帧索引，用作 PnP 初值
        if((frame_it->first) > Headers[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix(); // 这里的R_inital是到世界系(滑窗中第l帧坐标系)到滑窗中第i帧图像系的变换 R_C_W
        Vector3d P_inital = - R_inital * T[i]; // t_cw = -R_cw * t_wc
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec); //罗德里格斯公式将旋转矩阵转换成旋转向量
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;   // 存储3d路标点
        vector<cv::Point2f> pts_2_vector;   // 存储2d对应像素点，大小与pts_3_vector一样，且一一对应
        // map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>> > > points; 是特征提取的一个数据结构
        // 是当前帧(包括左右双目)所有图像特征点id与左右双目特征点对应信息的键值对，(feature_id, [camera_id, [x,y,z,u,v,vx,vy]])
        for (auto &id_pts : frame_it->second.points) // 遍历该图像帧的所有特征点，并构造3d-2d点的匹配对
        {
            int feature_id = id_pts.first; // 特征点id(唯一)
            for (auto &i_p : id_pts.second) // 遍历该特征点在各个相机下的观测(左右双目)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end()) // 如果该特征点在前面的初始化sfm过程中被成功三角化
                {
                    Vector3d world_pts = it->second; // 获取该特征点的3d路标点坐标(相对于参考帧l)
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>(); // 该特征点在当前帧下的相机归一化平面坐标
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);  // 因为是相机归一化平面坐标，所以这里的K是单位阵   
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        /*
        bool cv::solvePnP( 已知 3D 点的世界坐标和它们在图像上的 2D 像素坐标，求解相机的位姿（旋转和平移）
            InputArray objectPoints,   // 输入：世界坐标系下的 3D 点 (vector<Point3f>)
            InputArray imagePoints,    // 输入：图像平面上的 2D 点 (vector<Point2f>)，可以是像素坐标也可以是归一化平面坐标
            InputArray cameraMatrix,   // 输入：相机内参矩阵 K (3x3)，像素坐标系下需要传入真实内参，归一化平面坐标系下传单位阵
            InputArray distCoeffs,     // 输入：畸变系数 (如果点已经去畸变，这里传空)
            OutputArray rvec,          // 输出：旋转向量 (Rodrigues 向量, 描述世界坐标系如何转到相机坐标系，即 R_c_w)
            OutputArray tvec,          // 输出：平移向量 (描述世界原点在相机坐标系下的位置，即 t_c_w)
            bool useExtrinsicGuess = false, // 是否使用 rvec/tvec 的初始值进行优化
            int flags = SOLVEPNP_ITERATIVE  // 求解方法
        ); // objectPoints / imagePoints 必须一一对应，即objectPoints[i] 与 imagePoints[i] 是同一个特征点的 3D/2D
        点数下限  一般最少 4 个点（P3P 类方法）或 6 个点（DLT/迭代更稳），工程上建议 >20 且有足够视差/分布
        常用 flags:
            SOLVEPNP_ITERATIVE：迭代法（Levenberg-Marquardt 非线性优化）精度最高，速度慢，依赖初值          工程最佳场景：连续跟踪 (Tracking)
            SOLVEPNP_P3P：P3P 算法，需要且仅需要 4 个点，速度快，但对噪声敏感，且有多解，需要额外点来消除歧义    工程最佳场景：solvePnPRansac 的内核
            SOLVEPNP_EPNP：EPnP 算法，非迭代算法，速度快，精度一般,点多时常用作初值              工程最佳场景：全局重定位 / 掉线找回 / 初始化
            SOLVEPNP_IPPE / SOLVEPNP_IPPE_SQUARE: 这是二维码/Tag 识别专用的                工程最佳场景：AprilTag, Aruco, 二维码识别
        典型失败/误解来源:
            1. 3D 点尺度错/坐标系错（例如 SfM 点尺度不定但你当成米）;
            2. 2D 点未去畸变但你当成去畸变点了（或反之）;
            3. 点集中在小区域 / 近共面 → 退化;
            4. 外点没剔除（应考虑 solvePnPRansac）
        */
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose(); // R_w_c
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);   // t_w_c = -R_w_c * t_c_w
        frame_it->second.R = R_pnp * RIC[0].transpose(); // 转换到IMU body系下 R_w_i = R_w_c * R_c_i
        frame_it->second.T = T_pnp;
    }
    // 进行视觉-惯性联合初始化
    // 因为视觉SFM在初始化的过程中有着较好的表现，所以在初始化的过程中主要以SFM为主，然后将IMU的预积分结果与其对齐，即可得到较好的初始化结果
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

// 将之前建立的“纯视觉 SfM 地图”（任意尺度、任意方向）与 IMU 预积分进行融合，
// 解算出真实的物理尺度（Scale）、重力方向（Gravity）、初始速度（Velocity）和陀螺仪偏置（Gyro Bias），
// 并将整个系统转换到重力对齐的世界坐标系（Z轴垂直向上）
bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    // 将视觉SFM结果与IMU预积分结果进行对齐，
    // 得到陀螺仪偏置Bgs、在body坐标系下表示的每一IMU速度V^bn_bn、在cl帧坐标系下表示的重力向量g^cl、尺度因子s
    // 这里没有处理线加速度偏置，因为重力是初始化过程中的待求量，而线加速度计偏置与重力耦合；
    // 而且系统的线加速度相对于重力加速度很小，所以线加速度计偏置在初始化过程中很难观测，
    // 因此初始化过程中不考虑线加速度计偏置的估计
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++) // 遍历当前滑动窗口中的所有图像帧
    {
        Matrix3d Ri = all_image_frame[Headers[i]].R; // 该图像帧时对应的IMU坐标系到相机帧l坐标系的旋转变换
        Vector3d Pi = all_image_frame[Headers[i]].T; // 该图像帧到相机帧l坐标系的平移变换，即第l帧相机坐标系原点->i帧相机坐标系原点的平移向量，以第l帧坐标系为参考坐标系
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[Headers[i]].is_key_frame = true;
    }

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++) // 更新了陀螺仪的偏置Bgs之后，需要重新计算IMU预积分
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    f_manager.clearDepth();
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

    return true;
}

/**
 * @brief 在滑动窗口中寻找一个“最合适”的参考帧，与当前最新的帧进行对极几何求解，从而算出它们之间的相对旋转（R）和平移（T）
 * 这个相对位姿将作为后续 SfM（Structure from Motion）构建局部地图的**“种子”或“基准尺”**
 * 在初始化时，我们需要建立一个局部地图。但是，并不是随便找两帧就能算出可靠的位姿。
 *   - 如果两帧间隔太近（视差太小），三角化出来的深度误差极大，甚至会有负深度。
 *   - 如果共视点太少，解算本质矩阵（Essential Matrix）会很不稳定。
 * 因此，这个函数采用了一个“从远到近”的策略来筛选帧。
 * @param relative_R 当前最新帧到l帧之间的旋转矩阵R
 * @param relative_T 当前最新帧到l帧之间的平移矩阵
 * @param l 滑窗中最满足跟当前最新帧进行初始化的那一帧(l为在滑窗中的索引)
 * @return true 
 * @return false 
 */
bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++) // 从最老的一帧往新帧遍历
    {
        vector<pair<Vector3d, Vector3d>> corres; // (i, WINDOW_SIZE) 去畸变归一化平面坐标
        corres = f_manager.getCorresponding(i, WINDOW_SIZE); // 获取第 i 帧和最新帧（WINDOW_SIZE）之间的所有共视特征点
        // 共视特征点数量要大于一定的阈值， 20 是经验阈值，保证一定的鲁棒性余量
        // 求解对极几何（五点法或八点法）理论上只需要 5 或 8 个点，但在工程中，RANSAC 剔除外点后剩下的点往往不多，且存在噪声。
        if (corres.size() > 20)
        {
            // 计算视差
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm(); // 计算该特征点在两帧之间的移动距离（欧氏距离）
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size()); // 这里是归一化视差，没有单位，下面转换成像素级单位
            // 平均像素视差大于一定的值，这里460是焦距f的一个经验值
            // 利用共视特征点能够求解对极约束，解算出本质矩阵，并恢复R/T
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        if(USE_IMU)
        {
            para_SpeedBias[i][0] = Vs[i].x();
            para_SpeedBias[i][1] = Vs[i].y();
            para_SpeedBias[i][2] = Vs[i].z();

            para_SpeedBias[i][3] = Bas[i].x();
            para_SpeedBias[i][4] = Bas[i].y();
            para_SpeedBias[i][5] = Bas[i].z();

            para_SpeedBias[i][6] = Bgs[i].x();
            para_SpeedBias[i][7] = Bgs[i].y();
            para_SpeedBias[i][8] = Bgs[i].z();
        }
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }


    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }

    if(USE_IMU)
    {
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                          para_Pose[0][3],
                                                          para_Pose[0][4],
                                                          para_Pose[0][5]).toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        //TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5]).toRotationMatrix().transpose();
        }

        for (int i = 0; i <= WINDOW_SIZE; i++)
        {

            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) + origin_P0;


                Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                            para_SpeedBias[i][1],
                                            para_SpeedBias[i][2]);

                Bas[i] = Vector3d(para_SpeedBias[i][3],
                                  para_SpeedBias[i][4],
                                  para_SpeedBias[i][5]);

                Bgs[i] = Vector3d(para_SpeedBias[i][6],
                                  para_SpeedBias[i][7],
                                  para_SpeedBias[i][8]);
            
        }
    }
    else
    {
        for (int i = 0; i <= WINDOW_SIZE; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
            
            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if(USE_IMU)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            tic[i] = Vector3d(para_Ex_Pose[i][0],
                              para_Ex_Pose[i][1],
                              para_Ex_Pose[i][2]);
            ric[i] = Quaterniond(para_Ex_Pose[i][6],
                                 para_Ex_Pose[i][3],
                                 para_Ex_Pose[i][4],
                                 para_Ex_Pose[i][5]).normalized().toRotationMatrix();
        }
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    if(USE_IMU)
        td = para_Td[0][0];

}

bool Estimator::failureDetection()
{
    return false;
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5)
    {
        //ROS_INFO(" big translation");
        //return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        //ROS_INFO(" big z translation");
        //return true; 
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

void Estimator::optimization()
{
    TicToc t_whole, t_prepare;
    vector2double();

    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    //loss_function = NULL;
    loss_function = new ceres::HuberLoss(1.0);
    //loss_function = new ceres::CauchyLoss(1.0 / FOCAL_LENGTH);
    //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);
    for (int i = 0; i < frame_count + 1; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        if(USE_IMU)
            problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS);
    }
    if(!USE_IMU)
        problem.SetParameterBlockConstant(para_Pose[0]);

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization);
        if ((ESTIMATE_EXTRINSIC && frame_count == WINDOW_SIZE && Vs[0].norm() > 0.2) || openExEstimation)
        {
            //ROS_INFO("estimate extinsic param");
            openExEstimation = 1;
        }
        else
        {
            //ROS_INFO("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]);
        }
    }
    problem.AddParameterBlock(para_Td[0], 1);

    if (!ESTIMATE_TD || Vs[0].norm() < 0.2)
        problem.SetParameterBlockConstant(para_Td[0]);

    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks);
    }
    if(USE_IMU)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]);
            problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
    }

    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
 
        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]);
            }

            if(STEREO && it_per_frame.is_stereo)
            {                
                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)
                {
                    ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
                else
                {
                    ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]);
                }
               
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    //printf("prepare for ceres: %f \n", t_prepare.toc());

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    //printf("solver costs: %f \n", t_solver.toc());

    double2vector();
    //printf("frame_count: %d \n", frame_count);

    if(frame_count < WINDOW_SIZE)
        return;
    
    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD)
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        if(USE_IMU)
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if(imu_i != imu_j)
                    {
                        Vector3d pts_j = it_per_frame.point;
                        ProjectionTwoFrameOneCamFactor *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    if(STEREO && it_per_frame.is_stereo)
                    {
                        Vector3d pts_j_right = it_per_frame.pointRight;
                        if(imu_i != imu_j)
                        {
                            ProjectionTwoFrameTwoCamFactor *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{0, 4});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                        else
                        {
                            ProjectionOneFrameTwoCamFactor *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                           vector<double *>{para_Ex_Pose[0], para_Ex_Pose[1], para_Feature[feature_index], para_Td[0]},
                                                                                           vector<int>{2});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        marginalization_info->marginalize();
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            if(USE_IMU)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++)
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        if (last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
        
    }
    else
    {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize();
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++)
            {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if(USE_IMU)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            if (last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
            
        }
    }
    //printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
    //printf("whole time for ceres: %f \n", t_whole.toc());
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD)
    {
        double t_0 = Headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE)
        {
            for (int i = 0; i < WINDOW_SIZE; i++)
            {
                Headers[i] = Headers[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                if(USE_IMU)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];

            if(USE_IMU)
            {
                Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
                Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
                Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }

            if (true || solver_flag == INITIAL)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                all_image_frame.erase(all_image_frame.begin(), it_0);
            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE)
        {
            Headers[frame_count - 1] = Headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            if(USE_IMU)
            {
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    double tmp_dt = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }

                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[WINDOW_SIZE];
                pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

                dt_buf[WINDOW_SIZE].clear();
                linear_acceleration_buf[WINDOW_SIZE].clear();
                angular_velocity_buf[WINDOW_SIZE].clear();
            }
            slideWindowNew();
        }
    }
}

void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}


void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}

void Estimator::predictPtsInNextFrame()
{
    //printf("predict pts in next frame\n");
    if(frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion
    Eigen::Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count - 1, prevT);
    nextT = curT * (prevT.inverse() * curT);
    map<int, Eigen::Vector3d> predictPts;

    for (auto &it_per_id : f_manager.feature)
    {
        if(it_per_id.estimated_depth > 0)
        {
            int firstIndex = it_per_id.start_frame;
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
            //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
            if((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count)
            {
                double depth = it_per_id.estimated_depth;
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
                int ptsIndex = it_per_id.feature_id;
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    featureTracker.setPrediction(predictPts);
    //printf("estimator output %d predict pts\n",(int)predictPts.size());
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                 Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                 double depth, Vector3d &uvi, Vector3d &uvj)
{
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

void Estimator::outliersRejection(set<int> &removeIndex)
{
    //return;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
        feature_index ++;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;             
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                    Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                    depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            if(STEREO && it_per_frame.is_stereo)
            {
                
                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)
                {            
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }
                else
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0], 
                                                        Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                        depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", FOCAL_LENGTH / 1.5 * tmp_error);
                }       
            }
        }
        double ave_err = err / errCnt;
        if(ave_err * FOCAL_LENGTH > 3)
            removeIndex.insert(it_per_id.feature_id);

    }
}

void Estimator::fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity)
{
    double dt = t - latest_time;
    latest_time = t;
    Eigen::Vector3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    Eigen::Vector3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}

void Estimator::updateLatestStates()
{
    mPropagate.lock();
    latest_time = Headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    queue<pair<double, Eigen::Vector3d>> tmp_accBuf = accBuf;
    queue<pair<double, Eigen::Vector3d>> tmp_gyrBuf = gyrBuf;
    mBuf.unlock();
    while(!tmp_accBuf.empty())
    {
        double t = tmp_accBuf.front().first;
        Eigen::Vector3d acc = tmp_accBuf.front().second;
        Eigen::Vector3d gyr = tmp_gyrBuf.front().second;
        fastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mPropagate.unlock();
}
