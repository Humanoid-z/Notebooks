# Spring Boot自动配置

Spring Boot automatically:

- 启用Spring Security的默认配置，该配置创建一个Servlet过滤器，名为springSecurityFilterChain的bean。 该bean负责应用程序中的所有安全性（保护应用程序URL，验证提交的用户名和密码，重定向到登录表单等）。 
- 创建一个UserDetailsService bean，其中包含用户名user和随机生成的密码，该密码将记录到控制台。
-  针对每个请求，使用Servlet容器向名为springSecurityFilterChain的bean注册过滤器。

Spring Boot的配置不多，但功能很多。 功能摘要如下：

- 需要经过身份验证的用户才能与应用程序进行任何交互
- 为您生成默认的登录表单
- 让具有用户名user和密码的用户登录到控制台以使用基于表单的身份验证进行身份验证（在前面的示例中，密码为 8e557245-73e2-4286-969a-ff57fe326336）
- 使用BCrypt保护密码存储
- 让用户注销
- CSRF攻击防护
- 会话固定保护
- 安全标头集成
  - 用于安全请求的HTTP严格传输安全性
  - X-Content-Type-Options集成
  - 缓存控制（稍后可被您的应用程序覆盖以允许缓存您的静态资源）
  - X-XSS-Protection集成
  - X-Frame-Options集成有助于防止Clickjacking

- 与以下Servlet API方法集成：
  - HttpServletRequest#getRemoteUser()
  - HttpServletRequest.html#getUserPrincipal()
  - HttpServletRequest.html#isUserInRole(java.lang.String)
  - HttpServletRequest.html#login(java.lang.String,java.lang.String)
  - HttpServletRequest.html#logout()

# The Big Picture

## 过滤器的回顾

Spring Security的Servlet支持基于Servlet过滤器，因此通常首先了解过滤器的作用会很有帮助。下图显示了单个HTTP请求的处理程序的典型分层。

<img src="C:\Users\12548\Documents\GitHub\Notebooks\Spring Security\Untitled.assets\filterchain.png" alt="filterchain" style="zoom:67%;" />

客户端向应用程序发送请求，然后容器创建一个FilterChain，其中包含应该根据请求URI的路径处理HttpServletRequest的过滤器和Servlet。 在Spring MVC应用程序中，Servlet是DispatcherServlet的实例。 一个Servlet最多只能处理一个HttpServletRequest和HttpServletResponse。 但是，可以使用多个过滤器来：

- 防止下游过滤器或Servlet被调用。 在这种情况下，过滤器通常将编写HttpServletResponse。 
- 修改下游过滤器和Servlet使用的HttpServletRequest或HttpServletResponse。

过滤器的功能来自传递给它的FilterChain。

由于过滤器仅影响下游过滤器和Servlet，因此调用每个过滤器的顺序非常重要。

# java配置

Spring 3.1在Spring Framework中添加了对Java配置的常规支持。 自从Spring Security 3.2以来，已经存在Spring Security Java配置支持，该支持使用户可以轻松配置Spring Security，而无需使用任何XML。 如果您熟悉安全命名空间配置，则应该在它与安全Java配置支持之间找到很多相似之处。

第一步是创建我们的Spring Security Java配置。 该配置将创建一个称为springSecurityFilterChain的Servlet过滤器，该过滤器负责应用程序内的所有安全性（保护应用程序URL，验证提交的用户名和密码，重定向到登录表单等）。 您可以在下面找到Spring Security Java配置的最基本示例：

```java
import org.springframework.beans.factory.annotation.Autowired;

import org.springframework.context.annotation.*;
import org.springframework.security.config.annotation.authentication.builders.*;
import org.springframework.security.config.annotation.web.configuration.*;

@EnableWebSecurity
public class WebSecurityConfig {

    @Bean
    public UserDetailsService userDetailsService() {
        InMemoryUserDetailsManager manager = new InMemoryUserDetailsManager();
        manager.createUser(User.withDefaultPasswordEncoder().username("user").password("password").roles("USER").build());
        return manager;
    }
}
```

## HttpSecurity

到目前为止，我们的WebSecurityConfig仅包含有关如何验证用户身份的信息。 Spring Security如何知道我们要要求所有用户进行身份验证？ Spring Security如何知道我们要支持基于表单的身份验证？ 实际上，在后台调用了一个名为WebSecurityConfigurerAdapter的配置类。 它具有一种名为configure的方法，具有以下默认实现：

~~~java
protected void configure(HttpSecurity http) throws Exception {
    http
        .authorizeRequests(authorize -> authorize
            .anyRequest().authenticated()
        )
        .formLogin(withDefaults())
        .httpBasic(withDefaults());
}
~~~

上面的默认配置：

- 确保对我们应用程序的任何请求都需要对用户进行身份验证
- 允许用户使用基于表单的登录名进行身份验证
- 允许用户使用HTTP Basic身份验证进行身份验证

