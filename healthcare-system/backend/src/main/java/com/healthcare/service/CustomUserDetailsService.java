package com.healthcare.service;
import com.healthcare.repository.UserRepository;import lombok.RequiredArgsConstructor;import org.springframework.security.core.authority.SimpleGrantedAuthority;import org.springframework.security.core.userdetails.*;import org.springframework.stereotype.Service;
@Service @RequiredArgsConstructor public class CustomUserDetailsService implements UserDetailsService{ private final UserRepository repo;
 public UserDetails loadUserByUsername(String email){ var u=repo.findByEmail(email).orElseThrow(()->new UsernameNotFoundException("User not found")); return new org.springframework.security.core.userdetails.User(u.getEmail(),u.getPassword(), java.util.List.of(new SimpleGrantedAuthority("ROLE_"+u.getRole().name()))); }
}
