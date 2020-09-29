# 一、Android系统特点

内存和进程管理方面，Android有自己的运行时和虚拟机

- Android为了保证高优先级进程运行和正在与用户交互进程的响应速度，允许停止或终止正在运行的低优先级进程，以释放被占用的系统资源

- Android进程的优先级并不是固定的，而是根据进程是否在前台或是否与用户交互而不断变化的

- Android为组件定义了生命周期，并统一进行管理和控制

  

Android提供轻量级的进程间通讯机制Intent，使用跨进程组件通信和发送系统级广播成为可能

界面设计上，提供了丰富的界面控件

- 加快了用户界面的开发速度，保证了Android平台上的程序界面的一致性

- Android将界面设计与程序逻辑分离，使用XML文件对界面布局进行描述，有利于界面的修改和维护

  

Android提供Service作为无用户界面、长时间后台运行组件

- Service无需用户干预，可以长时间、稳定的运行，可为应用程序提供特定的后台功能

  

支持高效、快速的数据存储方式：

- SharedPreferences

- 文件存储

- 轻量级关系数据库SQLite

  

为了便于跨进程共享数据，Android提供了通用的共享数据接口ContentProvider

- 可以无需了解数据源、路径的情况下，对共享数据进行查询、添加、删除和更新等操作

# 二、Android四大基本组件

- Activity 活动
- Service 服务
- BroadcastReceiver 广播接收器
- ContentProvider 内容提供器

## Activity 活动

- 通俗地认为是用户界面，一个应用程序包含多个Activity
- 有生命周期
- 需要在AndroidManifest.xml中进行声明
- 通过堆栈来管理Activity

## Service 服务

- 不直接与用户交互，没有用户界面
- 能够长期在后台运行
- 比Activity的优先级高，不会被轻易终止
- 需要在AndroidManifest.xml中进行声明
- 两种启动方式
  - 绑定式（不求同生、但求同死）
  - 独立启动式

## BroadcastReceiver 广播接收器

- 实现全局监听，完成不同组件之间的通信
- 没有用户界面，但可启动Activity或用NotificationManager来通知用户
- 需要在AndroidManifest.xml中进行声明
- 用途：
  - 用户主动检查版本更新，有更新时发送广播，由Receiver接收，并以Dialog的方式提示用户更新
  - 服务器推送通知，淘宝、银行等的通知

## ContentProvider 内容提供器

- 支持多个应用程序的数据共享，是跨应用共享数据的唯一方法
- 应用场景
  - 一个短信接收应用A，将接收到的陌生短信的发信人添加到联系人管理应用B中
  - 方法1：A直接去操作B所记录的数据，比如SPS、文件、数据库等
  - 方法2：B通过ContentProvider暴露自己的数据操作接口，其他应用都可访问

# 三、Android应用结构分析

## Android SDK目录详解

**add-ons**：存放Google API，比如Google Maps

**build-tools**：存放各版本SDK编译工具

**docs**：离线开发者文档 Android SDK API

**extras**：扩展开发包，如HAXM加速

**platforms**：各版本SDK，如android-25

**platforms-tools**：各版本SDK通用工具，如adb、sqlite等

**samples**：各版本API样例

**skins**：Android模拟器的皮肤

**sources**：各版本SDK源码

**tools**：重要的工具，比如ant、ddms、logcat、emulator等

**system-images**：AVD模拟器映像文件

## 应用程序结构

- app：应用程序的源码和资源文件就放在这个module当中
- build：编译后的文件存放的位置，最终生成的apk文件就在这个目录中
- libs：添加的*.jar或*.so等文件存放的位置
- src：androidTest、test和main，main文件夹下又分为java和res两个文件夹，java文件夹下存放的是java源代码，res文件夹下存放的是资源文件
- AndroidManifest.xml 清单文件
  - 程序名称、图标、访问权限等整体属性
  - 四大组件

