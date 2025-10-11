<!DOCTYPE html>
<html lang="id">

<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>mitraskul.id</title>
    <link rel="icon" type="image/x-icon" href="{{ asset('favicon.ico') }}">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet" />
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" />
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
    <style>
        /* ============ Base ============ */
        body {
            font-family: 'Poppins', sans-serif;
            margin: 0;
            background-color: #fff;
            overflow: hidden;
        }

        a {
            text-decoration: none;
        }

        .container-fluid {
            padding: 0;
            margin: 0;
        }

        /* ============ Navbar ============ */
        .navbar {
            background-color: #ffffff;
            border-bottom: 1px solid #ddd;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .navbar .logo {
            width: 80px;
            margin-left: 30px;
        }

        .user-info {
            display: flex;
            align-items: center;
            margin-right: 30px;
        }

        .user-info span {
            margin-right: 1rem;
            font-weight: 600;
        }

        .user-info .profile-picture {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            object-fit: cover;
            border: 2px solid #c04055;
        }

        /* ============ Layout ============ */
        .main-wrapper {
            display: flex;
            height: calc(100vh - 80px);
        }

        .sidebar {
            font-family: 'Poppins', sans-serif;
            font-size: 18px;
            height: 100%;
            background-color: #eff9ff;
            position: relative;
            flex-shrink: 0;
        }

        .nav-link {
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            color: #495057;
            text-decoration: none;
            transition: background 0.2s, color 0.2s;
        }

        .nav-link.active {
            background-color: #e7f1ff;
            color: #0d6efd;
            font-weight: 600;
        }

        .nav-link i {
            font-size: 1.1rem;
        }

        .sidebar h5 {
            letter-spacing: 0.5px;
        }

        .logout {
            cursor: pointer;
        }

        .content {
            flex: 1;
            overflow-y: auto;
            background-color: #fff;
        }


        /* ============ Responsive ============ */
        @media (max-width: 768px) {
            .main-wrapper {
                flex-direction: column;
            }

            .sidebar {
                display: none;
            }

            .user-info {
                display: none;
                ;
            }

            .banner-images {
                flex-direction: column;
                align-items: flex-start;
            }

            .banner-images img {
                width: 100%;
            }

            .banner .illustration {
                display: none;
            }
        }

        .partner-footer-logo {
            height: 40px;
            object-fit: contain;
        }

        .footer ul {
            padding-left: 0;
        }

        .footer li {
            margin-bottom: 8px;
        }

        .footer .bg-opacity-50 {
            background-color: rgba(255, 255, 255, 0.75) !important;
            /* semi-transparent white for readability */
        }

        .container-footer {
            width: 100%;
            height: 100%;
            ;
        }

        .card {
            border: 1px solid #e2e8f0;
            background-color: #fff;
        }

        .card h6 {
            font-size: 1rem;
        }

        .card p {
            font-size: 0.85rem;
        }

        .rounded-4 {
            border-radius: 1rem;
        }

        small.text-muted {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .fw-semibold {
            font-weight: 500;
        }
    </style>
</head>

<body>
    <!-- Navbar -->
    <div class="navbar">
        <div class="col-lg-6 d-flex">
            <div class="col-lg-2">
                <img src="{{ url('img/logo.png') }}" alt="Logo" class="logo" width="" />
            </div>
        </div>
        <div class="user-info">
            <span>Halo, {{ $user->alumniSiswaProfile->nama_lengkap }}</span>
            <img src="{{ asset('storage/' . $user->alumniSiswaProfile->foto_profil) }}" alt="Profile"
                class="profile-picture" />
        </div>
        <button class="btn d-md-none" type="button" data-bs-toggle="offcanvas" data-bs-target="#offcanvasSidebar">
            <i class="bi bi-list fs-3"></i>
        </button>
    </div>

    <div class="offcanvas offcanvas-start" tabindex="-1" id="offcanvasSidebar">
        <div class="offcanvas-header">
            <h5 class="offcanvas-title">Menu</h5>
            <button type="button" class="btn-close" data-bs-dismiss="offcanvas"></button>
        </div>
        <div class="offcanvas-body">
            <a class="d-block mb-2 text-dark fw-semibold" href="{{ route('alumni-siswa.index') }}"><i
                    class="bi bi-house-door-fill me-2"></i>Beranda</a>
            <a class="d-block mb-2 text-dark fw-semibold" href="{{ route('alumni-siswa.sertifikasi') }}"><i
                    class="bi bi-patch-check-fill me-2"></i>Sertifikasi</a>
            <a class="d-block mb-2 text-dark fw-semibold" href="{{ route('alumni-siswa.loker') }}"><i
                    class="bi bi-briefcase-fill me-2"></i>Loker</a>
            <a class="d-block mb-2 text-dark fw-semibold" href="{{ route('alumni-siswa.pelatihan') }}"><i
                    class="bi bi-journal-text me-2"></i>Pelatihan</a>
            <a class="d-block mb-2 text-dark fw-semibold" href="{{ route('alumni-siswa.profile') }}"><i
                    class="bi bi-person-fill me-2"></i>Profil</a>
            <form id="logout-form" action="{{ route('logout') }}" method="POST">
                @csrf
                <a href="#" class="logout" onclick="event.preventDefault(); confirmLogout();">
                    <i class="bi bi-box-arrow-right me-2"></i>Logout</a>
            </form>
        </div>
    </div>

    <div class="main-wrapper">
        <div class="sidebar sidebar-main d-flex flex-column p-3" style="width: 250px;">


            <!-- Navigation Links -->
            <a class="nav-link d-flex align-items-center mb-2 {{ request()->routeIs('alumni-siswa.index') ? 'active' : 'text-secondary' }}"
                href="{{ route('alumni-siswa.index') }}">
                <i class="bi bi-house-door-fill me-2"></i> Beranda
            </a>
            <a class="nav-link d-flex align-items-center mb-2 {{ request()->routeIs('alumni-siswa.sertifikasi') ? 'active' : 'text-secondary' }}"
                href="{{ route('alumni-siswa.sertifikasi') }}">
                <i class="bi bi-patch-check-fill me-2"></i> Sertifikasi
            </a>
            <a class="nav-link d-flex align-items-center mb-2 {{ request()->routeIs('alumni-siswa.loker') ? 'active' : 'text-secondary' }}"
                href="{{ route('alumni-siswa.loker') }}">
                <i class="bi bi-briefcase-fill me-2"></i> Loker
            </a>
            <a class="nav-link d-flex align-items-center mb-2 {{ request()->routeIs('alumni-siswa.pelatihan') ? 'active' : 'text-secondary' }}"
                href="{{ route('alumni-siswa.pelatihan') }}">
                <i class="bi bi-journal-text me-2"></i> Pelatihan
            </a>
            <a class="nav-link d-flex align-items-center mb-4 {{ request()->routeIs('alumni-siswa.profile') ? 'active' : 'text-secondary' }}"
                href="{{ route('alumni-siswa.profile') }}">
                <i class="bi bi-person-fill me-2"></i> Profil
            </a>

            <!-- Logout -->
            <form id="logout-form" action="{{ route('logout') }}" method="POST" class="mt-auto">
                @csrf
                <a href="#" class="nav-link d-flex align-items-center text-danger fw-semibold mb-4"
                    onclick="event.preventDefault(); confirmLogout();">
                    <i class="bi bi-box-arrow-right me-2"></i> Logout
                </a>
            </form>
        </div>


        <div class="content">
            <div class="container py-5 px-3">
                <div class="row g-4">
                    <!-- Left: Profile Section -->
                    <div class="col-lg-8">
                        <div class="card p-4 rounded-4">
                            <div class="text-center mb-4">
                                <img src="{{ asset('storage/' . $user->alumniSiswaProfile->foto_profil) }}"
                                    alt="Foto Profil" class="rounded-circle border border-3"
                                    style="width: 160px; height: 160px; object-fit: cover;">
                                <h4 class="fw-bold mt-3 mb-1">{{ $user->alumniSiswaProfile->nama_lengkap }}</h4>
                                <p class="text-muted">{{ $user->alumniSiswaProfile->status }}</p>
                                <a href="{{ route('alumni-siswa.editProfile') }}"
                                    class="btn btn-outline-primary btn-sm px-4">Edit Profil</a>
                                <button type="button" class="btn btn-outline-secondary btn-sm px-4 ms-2"
                                    data-bs-toggle="modal" data-bs-target="#editAccountModal">
                                    Edit Akun
                                </button>
                            </div>
                            <hr>
                            <div class="row">
                                @php
                                    $profile = $user->alumniSiswaProfile;
                                @endphp
                                <div class="col-md-6 mb-3">
                                    <small class="text-muted">NIK</small>
                                    <div class="fw-semibold">{{ $profile->nik }}</div>
                                </div>
                                {{-- <div class="col-md-6 mb-3">
                                    <small class="text-muted">Kelas</small>
                                    <div class="fw-semibold">{{ $profile->kelas }}</div>
                                </div> --}}
                                <div class="col-md-6 mb-3">
                                    <small class="text-muted">Jurusan</small>
                                    <div class="fw-semibold">{{ $profile->jurusan_sekolah }}</div>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <small class="text-muted">Asal Sekolah</small>
                                    <div class="fw-semibold">{{ $profile->asal_sekolah }}</div>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <small class="text-muted">NPSN</small>
                                    <div class="fw-semibold">{{ $profile->npsn }}</div>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <small class="text-muted">Email</small>
                                    <div class="fw-semibold">{{ $user->email }}</div>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <small class="text-muted">No. Telepon</small>
                                    <div class="fw-semibold">{{ $user->no_hp }}</div>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <small class="text-muted">Jenis Kelamin</small>
                                    <div class="fw-semibold">{{ $profile->jenis_kelamin }}</div>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <small class="text-muted">Tanggal Lahir</small>
                                    <div class="fw-semibold">{{ $profile->tanggal_lahir }}</div>
                                </div>
                                <div class="col-12 mb-3">
                                    <small class="text-muted">Alamat</small>
                                    <div class="fw-semibold">{{ $profile->alamat }}</div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Right: Additional Info Panel -->
                    <div class="col-lg-4">
                        <div class="d-flex flex-column gap-4">
                            <!-- Sertifikasi -->
                            <div class="card shadow-sm p-3 rounded-4">
                                <h6 class="fw-bold mb-3 text-primary">
                                    <i class="bi bi-patch-check-fill me-2"></i>Sertifikasi
                                </h6>

                                <p class="text-muted mb-3">
                                    Telah mendaftar <strong>{{ $user->daftarSertifikasi->count() }}</strong>
                                    sertifikasi.
                                </p>
                                {{-- @forelse ($user->daftarSertifikasi as $pendaftaran)
                                    @php
                                        $sertifikasi = $pendaftaran->sertifikasi;
                                        $penyelenggara = $sertifikasi?->mitra?->mitraProfile?->nama_instansi ?? '-';
                                    @endphp
                                    <div class="mb-2">
                                        <strong>{{ $sertifikasi->judul_sertifikasi ?? '-' }}</strong>
                                        <p class="mb-0 text-muted small">{{ $penyelenggara }}</p>
                                    </div>
                                @empty
                                    <p class="text-muted">Belum ada sertifikasi yang didaftarkan.</p>
                                @endforelse --}}
                            </div>



                            <!-- Loker -->
                            <div class="card shadow-sm p-3 rounded-4">
                                <h6 class="fw-bold mb-3 text-primary"><i class="bi bi-briefcase-fill me-2"></i>Loker
                                </h6>
                                <p class="text-muted mb-3">
                                    Telah mendaftar <strong>{{ $user->daftarLoker->count() }}</strong>
                                    lowongan kerja.
                                </p>
                                {{-- @forelse ($user->daftarLoker as $pendaftaran)
                                    @php
                                        $loker = $pendaftaran->loker;
                                        $penyelenggara = $loker?->mitra?->mitraProfile?->nama_instansi ?? '-';
                                    @endphp
                                    <div class="mb-2">
                                        <strong>{{ $loker->posisi ?? '-' }}</strong>
                                        <p class="mb-0 text-muted small">{{ $penyelenggara }}</p>
                                    </div>
                                @empty
                                    <p class="text-muted">Belum ada melamar lowongan kerja.</p>
                                @endforelse --}}
                            </div>

                            <!-- Pelatihan -->
                            <div class="card shadow-sm p-3 rounded-4">
                                <h6 class="fw-bold mb-3 text-primary"><i class="bi bi-journal-text me-2"></i>Pelatihan
                                </h6>
                                <p class="text-muted mb-3">
                                    Telah mendaftar <strong>{{ $user->daftarPelatihan->count() }}</strong>
                                    pelatihan.
                                </p>
                                {{-- @forelse ($user->daftarPelatihan as $pendaftaran)
                                    @php
                                        $pelatihan = $pendaftaran->pelatihan;
                                        $penyelenggara = $pelatihan?->mitra?->mitraProfile?->nama_instansi ?? '-';
                                    @endphp
                                    <div class="mb-2">
                                        <strong>{{ $pelatihan->nama_pelatihan ?? '-' }}</strong>
                                        <p class="mb-0 text-muted small">{{ $penyelenggara }}</p>
                                    </div>
                                @empty
                                    <p class="text-muted">Belum ada sertifikasi yang didaftarkan.</p>
                                @endforelse --}}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Modal Edit Akun -->
            <div class="modal fade" id="editAccountModal" tabindex="-1" aria-labelledby="editAccountModalLabel"
                aria-hidden="true">
                <div class="modal-dialog">
                    <form method="POST" action="{{ route('alumni-siswa.account.update') }}">
                        @csrf
                        @method('PUT')

                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title" id="editAccountModalLabel">Edit Akun</h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal"
                                    aria-label="Tutup"></button>
                            </div>
                            <div class="modal-body">
                                <!-- No HP -->
                                <div class="mb-3">
                                    <label for="no_hp" class="form-label">No. HP</label>
                                    <input type="text" name="no_hp" id="no_hp"
                                        value="{{ old('no_hp', $user->no_hp) }}" class="form-control" readonly>
                                    @error('no_hp')
                                        <div class="text-danger small">{{ $message }}</div>
                                    @enderror
                                </div>

                                <!-- Email -->
                                <div class="mb-3">
                                    <label for="email" class="form-label">Email Baru</label>
                                    <input type="email" name="email" id="email"
                                        value="{{ old('email', $user->email) }}" class="form-control" readonly>
                                    @error('email')
                                        <div class="text-danger small">{{ $message }}</div>
                                    @enderror
                                </div>

                                <!-- Password Baru -->
                                <div class="mb-3">
                                    <label for="password" class="form-label">Password Baru</label>
                                    <input type="password" name="password" id="password" class="form-control"
                                        autocomplete="new-password" required>
                                    @error('password')
                                        <div class="text-danger small">{{ $message }}</div>
                                    @enderror
                                </div>

                                <!-- Konfirmasi Password -->
                                <div class="mb-3">
                                    <label for="password_confirmation" class="form-label">Konfirmasi Password</label>
                                    <input type="password" name="password_confirmation" id="password_confirmation"
                                        class="form-control" autocomplete="new-password" required>
                                    @error('password_confirmation')
                                        <div class="text-danger small">{{ $message }}</div>
                                    @enderror
                                </div>
                            </div>
                            <div class="modal-footer">
                                <button type="button" class="btn btn-secondary"
                                    data-bs-dismiss="modal">Batal</button>
                                <button type="submit" class="btn btn-primary">Simpan Perubahan</button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>

            <footer class="footer text-dark"
                style="background: url('{{ url('img/footer.png') }}') no-repeat center center / cover;">
                <div class="container-footer">
                    <div class="row align-items-start bg-white bg-opacity-75 rounded-3 p-4 shadow-sm">
                        <!-- Logo & Deskripsi -->
                        <div class="col-md-4 mb-4">
                            <img src="{{ url('img/logo.png') }}" alt="Logo Skul.Id" style="height: 65px;"
                                class="mb-3">
                            <p class="small">
                                mitraskul.Id adalah platform yang menghubungkan alumni dengan sekolah, dunia industri,
                                dan
                                peluang karier.
                            </p>
                        </div>

                        <!-- Kontak -->
                        <div class="col-md-4 mb-4">
                            <h6 class="fw-bold text-uppercase mb-3">Kontak</h6>
                            <ul class="list-unstyled small">
                                <li><i class="bi bi-envelope-fill me-2"></i>mitraskulid@gmail.com</li>
                                <li><i class="bi bi-telephone-fill me-2"></i>+62 851-7959-2408</li>
                            </ul>
                        </div>

                        <!-- Partner -->
                        <div class="col-md-4 mb-4">
                            <h6 class="fw-bold text-uppercase mb-3">Didukung Oleh</h6>
                            <div class="d-flex align-items-center gap-3 flex-wrap">
                                <img src="{{ url('img/Telkomsel.png') }}" alt="Telkomsel"
                                    class="partner-footer-logo">
                                <img src="{{ url('img/pu.png') }}" alt="PUP Makassar" class="partner-footer-logo">
                            </div>
                        </div>
                        <div class="text-center text-dark small mt-4">
                            © 2025 Skul.Id. All rights reserved.
                        </div>
                    </div>

                </div>
            </footer>

            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
            <script>
                @if (session('success'))
                    Swal.fire({
                        icon: 'success',
                        title: 'Berhasil!',
                        text: {!! json_encode(session('success')) !!},
                        timer: 3000,
                        showConfirmButton: false
                    });
                @endif

                @if (session('error'))
                    Swal.fire({
                        icon: 'error',
                        title: 'Gagal!',
                        text: {!! json_encode(session('error')) !!},
                        timer: 3000,
                        showConfirmButton: false
                    });
                @endif

                @if (session('message'))
                    Swal.fire({
                        icon: '{{ session('alert-type') == 'warning' ? 'warning' : 'info' }}',
                        title: '{{ ucfirst(session('alert-type') ?? 'Info') }}',
                        text: {!! json_encode(session('message')) !!},
                        timer: 3000,
                        showConfirmButton: false
                    });
                @endif
            </script>
            <script>
                @if ($errors->any())
                    var editAccountModal = new bootstrap.Modal(document.getElementById('editAccountModal'));
                    editAccountModal.show();
                @endif
            </script>
            <script>
                function confirmLogout() {
                    event.preventDefault();
                    Swal.fire({
                        title: 'Yakin ingin logout?',
                        text: "Anda akan keluar dari akun ini.",
                        icon: 'warning',
                        showCancelButton: true,
                        confirmButtonColor: '#d33',
                        cancelButtonColor: '#6c757d',
                        confirmButtonText: 'Ya, logout!',
                        cancelButtonText: 'Batal'
                    }).then((result) => {
                        if (result.isConfirmed) {
                            document.getElementById('logout-form').submit();
                        }
                    });
                }
            </script>

</body>

</html>
