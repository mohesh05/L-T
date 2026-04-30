package com.healthcare.exception;
import org.springframework.http.*;import org.springframework.web.bind.MethodArgumentNotValidException;import org.springframework.web.bind.annotation.*;import java.util.Map;
@RestControllerAdvice public class GlobalExceptionHandler {
@ExceptionHandler(ApiException.class) public ResponseEntity<?> api(ApiException e){ return ResponseEntity.badRequest().body(Map.of("error",e.getMessage())); }
@ExceptionHandler(MethodArgumentNotValidException.class) public ResponseEntity<?> valid(MethodArgumentNotValidException e){ return ResponseEntity.status(HttpStatus.UNPROCESSABLE_ENTITY).body(Map.of("error","Validation failed")); }
}
