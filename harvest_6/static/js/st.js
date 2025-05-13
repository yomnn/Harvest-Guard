TweenMax.staggerFrom(".form-group", 1, {
  delay: 0.2,
  opacity: 0,
  y: 20,
  ease: Expo.easeInOut
}, 0.2);

TweenMax.staggerFrom(".contact-info-container > *", 1, {
  delay: 0,
  opacity: 0,
  y: 20,
  ease: Expo.easeInOut
}, 0.1);
function validateForm() {
    var name = document.getElementById('name').value;
    var email = document.getElementById('email').value;
    var message = document.getElementById('message').value;

    // Reset error messages
    document.getElementById('nameError').innerText = '';
    document.getElementById('emailError').innerText = '';
    document.getElementById('messageError').innerText = '';

    if (name === '') {
      document.getElementById('nameError').innerText = 'Please enter your name';
      return false;
    }

    if (email === '') {
      document.getElementById('emailError').innerText = 'Please enter your email';
      return false;
    } else if (!isValidEmail(email)) {
      document.getElementById('emailError').innerText = 'Please enter a valid email address';
      return false;
    }

    if (message === '') {
      document.getElementById('messageError').innerText = 'Please enter your message';
      return false;
    }

    // Add more complex validation logic if needed

    return true;
  }

  function isValidEmail(email) {
    // Add your email validation logic here
    var emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);}
