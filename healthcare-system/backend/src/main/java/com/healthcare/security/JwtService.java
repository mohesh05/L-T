package com.healthcare.security;
import io.jsonwebtoken.*;import io.jsonwebtoken.security.Keys;import org.springframework.beans.factory.annotation.Value;import org.springframework.stereotype.Service;import javax.crypto.SecretKey;import java.util.*;
@Service public class JwtService {
 @Value("${app.jwt.secret}") private String secret; @Value("${app.jwt.expiration}") private long expiration;
 private SecretKey key(){ return Keys.hmacShaKeyFor(secret.getBytes()); }
 public String generate(String username,String role){ return Jwts.builder().subject(username).claim("role",role).issuedAt(new Date()).expiration(new Date(System.currentTimeMillis()+expiration)).signWith(key()).compact(); }
 public String extractUsername(String token){ return Jwts.parser().verifyWith(key()).build().parseSignedClaims(token).getPayload().getSubject(); }
 public boolean isValid(String token){ try{ Jwts.parser().verifyWith(key()).build().parseSignedClaims(token); return true;}catch(Exception e){return false;} }
}
