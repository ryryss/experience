# vulkan经验总结

## 学习文档
vulkan https://zhuanlan.zhihu.com/p/165341403
<br><br>
强推intel vulkan utorial，看这个链接就行，不需要往下看我写的东西了

https://www.intel.com/content/www/us/en/developer/articles/training/api-without-secrets-introduction-to-vulkan-part-1.html

例如，Selecting Presentation Mode这一节，就很好的介绍了需要3缓冲的原因，以及送显模式如何平衡一些问题
<br><br>

## vulkan初始化和执行流程、数据关联情况
### 初始化 静态部分
实例(VkInstance)<br>
最先初始化的一定是VkInstance，在创建信息(VkInstanceCreateInfo)中，
需要指定扩展(ppEnabledExtensionNames)和层(ppEnabledLayerNames)
<br><br>
物理设备(VkPhysicalDevice)<br>
通过VkInstance枚举(vkEnumeratePhysicalDevices)获得
固有内容，只能让硬件工程师或者驱动工程师改动
<br><br>
逻辑设备(VkDevice)<br>
在创建信息(VkInstanceCreateInfo)中需要提供队列创建信息(pQueueCreateInfos = VkDeviceQueueCreateInfo[])
需要指定扩展(ppEnabledExtensionNames)<br>
最好把硬件功能也打开(pEnabledFeatures = vkGetPhysicalDeviceFeatures)<br>
1个物理设备可以划分n个逻辑设备<br><br>

### 以下内容都基于逻辑设备，可认为是逻辑设备拥有的资源，对应的创建函数基本都需要一个device入参

队列(VkQueue)<br>
有多少条VkQueue由VkPhysicalDevice决定(vkGetPhysicalDeviceQueueFamilyProperties)
在创建信息(VkDeviceQueueCreateInfo)中设置VkDevice队列的数量(queueCount)
<br><br>
交换链(VkSwapchainKHR)<br>
通常情况下，每个窗口(VkSurfaceKHR)对应一个VkSwapchainKHR，本例的窗口来源是glfw三方库(glfwCreateWindowSurface)
在创建信息(VkSwapchainCreateInfoKHR)中可以设定缓存图像的数量(minImageCount)
但最大最小的缓存图像数量，是由VkPhysicalDevice决定的(vkGetPhysicalDeviceSurfaceCapabilitiesKHR)
(这里返回了软硬件混合信息，设计得很不好)<br>
minImageCount不能超过VkSurfaceCapabilitiesKHR.maxImageCount
<br><br>
图像(VkImage)和视图(VkImageView)<br>
VkImage的来源是vkGetSwapchainImagesKHR，就是VkSwapchainKHR上的图像
VkImageView是VkImage的容器和类型描述，在填创建信息(imageViewCreateInfo)时绑定image
注意，这里的描述不涉及图像的宽高，图像的宽高由VkSurfaceKHR绑定的窗口(window)决定
<br><br>
命令池(VkCommandPool)和命令buff(VkCommandBuffer)<br>
VkCommandPool生成VkCommandBuffer，后者存储着一些命令
命令的是cpu发给gpu的(可能在驱动层有转换，但用图形库的应用角度来说，等价于如此)
有些代码cpu跑得实在是太慢了，让gpu跑更好
<br><br>
渲染通道(VkRenderPass) 子通道(VkSubpass) 帧缓冲(VkFramebuffer) 附件(Attachment)<br>
渲染通道玩意的定义，包括VkSubpass，我目前都没怎么搞明白，只知道它叫渲染过程，我打算用反推的方式来敲定它的作用
两者的创建信息(VkRenderPassCreateInfo、VkSubpassDescription)中都涉及Attachment
VkSubpassDescription中可同时持有多种VkAttachmentReference
在VkFramebuffer的创建信息(framebufferCreateInfo)中，绑定VkImageView，并要求输入宽高，还要求输入VkRenderPass

而谷歌到Attachment的作用是作为渲染过程的中间暂存资源，Attachment的来源可能是开发者自己输入，上一个VkSubpass的输出，或者就是图像等
在创建信息(VkAttachmentDescription)中的2个Layout变量，就是用于标记这个Attachment属于什么类型的Attachment
(这里有啥毛病，，，用2个变量标记，头痛死，就不能搞个枚举清楚的说明这个Attachment是个啥)
所以反推出VkRenderPass的作用是，通过VkFramebuffer取VkImage的一块区域，并使用attachment对其进行改动!!!
(有的文章说VkFramebuffer也是一个图像附件，这样理解确实舒服多了)
<br><br>
管线布局(VkPipelineLayout) 管线(VkPipeline) 渲染模块(VkShaderModule) VkPushConstantRange(常量范围)<br>
VkShaderModule使用glslc编译而成，具体使用读取方式，见源码(对官方方案有简单改动)<br>
VkPushConstantRange最大值能到100+KB，本例只需要1个float传递角度参数<br>
VkPipelineLayout在本例的作用是把VkPushConstantRange送到VkShaderModule<br>
shader的输入还由VkPipeline的创建信息(VkGraphicsPipelineCreateInfo)里的变量pVertexInputState控制<br>
至于VkPipeline，它的创建信息(VkGraphicsPipelineCreateInfo)需要一个VkRenderPass输入
<br><br>
VkBuffer(一段连续内存) VkDeviceMemory(设备内存)
<br><br>
VkDeviceMemory是设备端的一块内存(显存)，申请前需要使用vkGetBufferMemoryRequirements取得real size(可能比预期的size大)
使用vkMapMemory获得将设备内存映射到一个空指针，再使用memcpy就能读写device内存
<br><br>
VkBuffer是一个buff对象，在其创建信息(VkBufferCreateInfo)中需要配置队列信息和用途
使用vkCmdBindVertexBuffers将VkBuffer和VkDeviceMemory绑定，VkBuffer才能使用
VkBuffer在本例的用途是传递顶点数据vkCmdBindVertexBuffers

还有一个概念是可见性：host(device)是否能映射并读写对方的内存，这里暂不展开
<br><br>
VkSemaphore(信号量) VkFence(栅栏)
<br><br>
VkFence用于host(cpu)和device(gpu)同步
<br><br>
VkSemaphore用于VkQueue同步
还有其他的同步原语 没用到就先不管了
<br><br>
### 渲染循环 动态部分
vkBeginCommandBuffer，VkCommandBuffer开始录制<br>
vkCmdBeginRenderPass，录制的命令(入参1)会应用于此VkRenderPass，创建信息<br>(VkRenderPassBeginInfo)可配置渲染区域(renderArea)等
(简单遗留，1个VkSubpass到底只跑1个shader还是所有shader都跑1趟)<br>
vkCmdBindPipeline，录制的命令会应用于此VkPipeline<br>
vkCmdBindVertexBuffers，绑定顶点输入(这里我超级想吐槽，但是又能理解为啥这么搞)<br>
vkCmdPushConstants，把angle变量送到shader<br>
vkCmdDraw，绘制 可以一直加命令，加加加到厌倦<br>
vkCmdEndRenderPass，渲染结束<br>
vkEndCommandBuffer，结束录制<br>
vkQueueSubmit，提交到某个VkQueue<br>
vkQueuePresentKHR，使用VkQueue送显<br>
vkWaitForFences，等待gpu渲染完成<br>
vkResetFences，fence重置为无信号，表示我暂时不需要用<br><br>

## 最后总结
1、总的来说，vulkan就是一套使用图形后端的接口，使用各种vulkan函数去驱动图像后端，开发者接到一个需求，要知道怎么调vulkan的接口去完成

2、英特尔的资料的资料写得真的很好

3、这个文档对应的源码丢在华为的电脑了，是一个最小的vulkan可运行例子，没有任何的错误处理