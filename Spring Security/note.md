# Spring Boot自动配置

Spring Boot automatically:

启用Spring Security的默认配置，该配置创建一个Servlet过滤器，名为springSecurityFilterChain的bean。 该bean负责应用程序中的所有安全性（保护应用程序URL，验证提交的用户名和密码，重定向到登录表单等）。 

创建一个UserDetailsService bean，其中包含用户名user和随机生成的密码，该密码将记录到控制台。

 针对每个请求，使用Servlet容器向名为springSecurityFilterChain的bean注册过滤器。