import { useState } from 'react';
import { View, StyleSheet, ScrollView, KeyboardAvoidingView, Platform } from 'react-native';
import { TextInput, Button, Text, HelperText, Divider } from 'react-native-paper';
import { Link, router } from 'expo-router';
import { useAuthStore } from '@/store/authStore';

export default function LoginScreen() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [emailError, setEmailError] = useState('');

  const { signIn, signInWithGoogle, signInWithApple, isLoading, error, clearError } =
    useAuthStore();

  const validateEmail = (email: string) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  const handleSignIn = async () => {
    clearError();
    setEmailError('');

    if (!validateEmail(email)) {
      setEmailError('Please enter a valid email address');
      return;
    }

    if (password.length < 6) {
      return;
    }

    try {
      await signIn(email, password);
      router.replace('/(tabs)');
    } catch (err) {
      // Error is handled by store
    }
  };

  return (
    <KeyboardAvoidingView
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      style={styles.container}
    >
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text variant="displaySmall" style={styles.logo}>
            📚 BookFlow
          </Text>
          <Text variant="headlineSmall" style={styles.title}>
            Welcome Back
          </Text>
          <Text variant="bodyLarge" style={styles.subtitle}>
            Sign in to continue
          </Text>
        </View>

        <View style={styles.form}>
          <TextInput
            label="Email"
            value={email}
            onChangeText={text => {
              setEmail(text);
              setEmailError('');
            }}
            keyboardType="email-address"
            autoCapitalize="none"
            autoComplete="email"
            error={!!emailError}
            disabled={isLoading}
            mode="outlined"
            style={styles.input}
          />
          <HelperText type="error" visible={!!emailError}>
            {emailError}
          </HelperText>

          <TextInput
            label="Password"
            value={password}
            onChangeText={setPassword}
            secureTextEntry={!showPassword}
            autoCapitalize="none"
            autoComplete="password"
            disabled={isLoading}
            mode="outlined"
            style={styles.input}
            right={
              <TextInput.Icon
                icon={showPassword ? 'eye-off' : 'eye'}
                onPress={() => setShowPassword(!showPassword)}
              />
            }
          />

          <Link href="/(auth)/forgot-password" style={styles.forgotPassword}>
            <Text variant="bodyMedium">Forgot Password?</Text>
          </Link>

          {error && (
            <HelperText type="error" visible style={styles.error}>
              {error}
            </HelperText>
          )}

          <Button
            mode="contained"
            onPress={handleSignIn}
            loading={isLoading}
            disabled={isLoading}
            style={styles.signInButton}
          >
            Sign In
          </Button>

          <Divider style={styles.divider} />
          <Text variant="bodyMedium" style={styles.orText}>
            OR
          </Text>

          <Button
            mode="outlined"
            onPress={() => signInWithGoogle()}
            disabled={isLoading}
            style={styles.socialButton}
            icon="google"
          >
            Continue with Google
          </Button>

          <Button
            mode="outlined"
            onPress={() => signInWithApple()}
            disabled={isLoading}
            style={styles.socialButton}
            icon="apple"
          >
            Continue with Apple
          </Button>

          <View style={styles.signUpContainer}>
            <Text variant="bodyMedium">Don't have an account? </Text>
            <Link href="/(auth)/signup">
              <Text variant="bodyMedium" style={styles.signUpLink}>
                Sign Up
              </Text>
            </Link>
          </View>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    padding: 24,
  },
  header: {
    alignItems: 'center',
    marginBottom: 32,
  },
  logo: {
    marginBottom: 8,
  },
  title: {
    marginBottom: 4,
  },
  subtitle: {
    color: '#757575',
  },
  form: {
    width: '100%',
    maxWidth: 400,
    alignSelf: 'center',
  },
  input: {
    marginBottom: 8,
  },
  forgotPassword: {
    alignSelf: 'flex-end',
    marginBottom: 16,
  },
  signInButton: {
    marginTop: 8,
    paddingVertical: 6,
  },
  divider: {
    marginVertical: 24,
  },
  orText: {
    textAlign: 'center',
    marginTop: -36,
    marginBottom: 24,
    backgroundColor: '#fff',
    alignSelf: 'center',
    paddingHorizontal: 16,
  },
  socialButton: {
    marginBottom: 12,
  },
  signUpContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 24,
  },
  signUpLink: {
    color: '#1976D2',
    fontWeight: '600',
  },
  error: {
    marginBottom: 8,
  },
});
