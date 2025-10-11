<!-- register.blade.php (versi yang disamakan dengan login) -->
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>mitraskul.id</title>
    <link rel="icon" type="image/x-icon" href="{{ asset('favicon.ico') }}">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet" />
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.5/dist/css/bootstrap.min.css" rel="stylesheet" />
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css" rel="stylesheet" />
    <style>
        html,
        body {
            height: 100%;
            margin: 0;
            padding: 0;
            font-family: "Poppins", sans-serif;
            background-color: #fff;
            color: #252f35;
            font-size: 16px;
        }

        .full-page-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 100%;
            padding: 40px;
            position: relative;
        }

        .btn-back {
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 999;
        }

        .logo-wrapper {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 40px;
        }

        .logo-wrapper img {
            height: 50px;
        }

        .inner-content {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 100px;
            flex-wrap: wrap;
            width: 100%;
            max-width: 1000px;
        }

        .icon-besar {
            flex: 1;
            min-width: 300px;
            text-align: center;
        }

        .icon-besar img {
            max-width: 100%;
            height: auto;
        }

        .register-form {
            flex: 1;
            min-width: 300px;
        }

        h1 {
            font-weight: 600;
            font-size: 36px;
            margin-bottom: 15px;
        }

        p {
            font-size: 18px;
        }

        .form-control {
            background-color: #f1f8f7;
            border: none;
            padding-left: 55px;
            height: 70px;
            font-size: 18px;
        }

        .form-control:focus {
            box-shadow: none;
        }

        .input-group-text {
            background: none;
            border: none;
            position: absolute;
            left: 15px;
            top: 50%;
            transform: translateY(-50%);
            color: #999;
            font-size: 22px;
            z-index: 2;
        }

        .form-check-input {
            width: 20px;
            height: 20px;
            margin-top: 5px;
        }

        .form-check-label {
            font-size: 16px;
        }

        .form-check a {
            text-decoration: underline;
            font-weight: 500;
        }

        .form-check-input:checked {
            background-color: #ff3c6a;
            border-color: #ff3c6a;
        }

        .login-btn {
            background: #00ffe1;
            color: white;
            font-weight: 600;
            font-size: 20px;
            height: 70px;
        }

        .alert {
            font-size: 16px;
        }

        @media (max-width: 768px) {
            .inner-content {
                flex-direction: column;
                text-align: center;
            }

            .icon-besar img {
                width: 300px;
            }

            h1 {
                font-size: 30px;
            }

            p,
            .form-check-label {
                font-size: 16px;
            }

            .form-control,
            .login-btn {
                font-size: 16px;
                height: 60px;
            }

            .full-page-container {
                padding: 20px;
            }

            .logo-wrapper {
                justify-content: center;
                padding: 0 20px;
                /* jarak sisi kiri/kanan */
                margin-bottom: 30px;
            }

            .btn-back {
                top: 15px;
                left: 15px;
                /* lebih menjauh dari tepi layar */
            }

            button {
                margin-bottom: 80px;
            }

            .log-logo {
                margin-top: 50px;
            }
        }

        .login-btn:hover {
            background-color: #00FFE1 !important;
            color: white !important;
            box-shadow: none !important;
            opacity: 1 !important;
            transform: none !important;
        }
    </style>
</head>

<body>
    <a href="{{ url()->previous() }}" class="btn-back">
        <img src="{{ url('img/back.png') }}" alt="Back" width="30" />
    </a>

    <div class="full-page-container">
        <div class="inner-content">
            <div class="icon-besar">
                <img src="{{ url('img/login-alumni.png') }}" class="log.logo" alt="Icon Besar" />
            </div>

            <div class="register-form">
                <div class="text mb-3">
                    <h1>Daftar Alumni</h1>
                    <p>Sudah memiliki akun? <a href="{{ route('login.alumni') }}" class="text-danger">Masuk di sini</a>
                    </p>
                </div>

                <form method="POST" action="{{ route('register.store') }}">
                    @csrf
                    <div class="mb-3 position-relative">
                        <span class="input-group-text"><i class="bi bi-phone"></i></span>
                        <input type="text" class="form-control" placeholder="Nomor HP Telkomsel" name="no_hp" required>
                        @if ($errors->has('no_hp'))
                            <div class="alert alert-danger mt-2">
                                {{ $errors->first('no_hp') }}
                            </div>
                        @endif
                    </div>

                    <div class="mb-3 position-relative">
                        <span class="input-group-text"><i class="bi bi-lock"></i></span>
                        <input type="password" class="form-control" placeholder="Password" name="password" minlength="8"
                            required>
                    </div>

                    <div class="mb-3 position-relative">
                        <span class="input-group-text"><i class="bi bi-lock-fill"></i></span>
                        <input type="password" class="form-control" placeholder="Konfirmasi Password"
                            name="password_confirmation" minlength="8" required>
                    </div>

                    <input type="hidden" name="role" value="alumnisiswa" />

                    <div class="form-check mb-3 text-start">
                        <input class="form-check-input" type="checkbox" id="termsCheck" required />
                        <label class="form-check-label" for="termsCheck">
                            Dengan mendaftar, saya menyetujui
                            <a href="#">Ketentuan Pengguna</a> & <a href="#">Kebijakan Privasi</a>
                        </label>
                    </div>

                    <button type="submit" class="btn login-btn w-100">DAFTAR</button>
                </form>
            </div>
        </div>
    </div>

    <script>
        document.querySelector("form").addEventListener("submit", function (e) {
            const phoneInput = document.querySelector('input[name="no_hp"]');
            const passwordInput = document.querySelector('input[name="password"]');

            const phonePattern = /^08(11|12|13|21|22|51|52|53)[0-9]{5,8}$/;
            const phoneValue = phoneInput.value;
            const passwordValue = passwordInput.value;

            let errorMessages = [];

            if (!phonePattern.test(phoneValue)) {
                errorMessages.push("Nomor HP harus nomor Telkomsel.");
            }

            if (passwordValue.length < 8) {
                errorMessages.push("Password minimal 8 karakter.");
            }

            if (errorMessages.length > 0) {
                e.preventDefault();
                alert(errorMessages.join("\n"));
            }
        });

        window.addEventListener("DOMContentLoaded", function () {
            const alerts = document.querySelectorAll(".alert");
            alerts.forEach(alert => {
                setTimeout(() => {
                    alert.style.transition = "opacity 0.5s ease";
                    alert.style.opacity = "0";
                    setTimeout(() => alert.remove(), 500);
                }, 3000);
            });
        });
    </script>
</body>

</html>