package com.healthcare.controller;
import com.healthcare.dto.AuthDtos.*;import com.healthcare.service.impl.AppService;import jakarta.validation.Valid;import lombok.RequiredArgsConstructor;import org.springframework.http.HttpStatus;import org.springframework.web.bind.annotation.*;
@RestController @RequestMapping("/api/auth") @RequiredArgsConstructor
public class AuthController { private final AppService s;
@PostMapping("/register") @ResponseStatus(HttpStatus.CREATED) public AuthRes register(@Valid @RequestBody RegisterReq r){return s.register(r);} @PostMapping("/login") public AuthRes login(@Valid @RequestBody LoginReq r){return s.login(r);} }
