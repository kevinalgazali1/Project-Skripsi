<?php

namespace App\Http\Controllers;

use App\Models\User;
use App\Models\Loker;
use App\Models\DaftarLoker;
use App\Models\DaftarPelatihan;
use App\Models\Sertifikasi;
use Illuminate\Http\Request;
use App\Models\DaftarSertifikasi;
use App\Models\Pelatihan;
use Illuminate\Support\Facades\Auth;

class DaftarController extends Controller
{
    public function storeDaftarSertifikasi(Request $request, Sertifikasi $sertifikasi)
    {
        try {
            $user = Auth::user(); // user yang sedang login
            $profile = $user->alumniSiswaProfile;

            if (!$profile) {
                return redirect()->back()->with('error', 'Data profil alumni belum lengkap.');
            }

            // Cek jika sudah pernah mendaftar sertifikasi
            $alreadyRegistered = DaftarSertifikasi::where('user_id', $user->id)
                ->where('sertifikasi_id', $sertifikasi->id)
                ->exists();

            if ($alreadyRegistered) {
                return redirect()->back()->with([
                    'message' => 'Anda sudah pernah mendaftar sertifikasi.',
                    'alert-type' => 'warning',
                ]);
            }

            // Validasi bukti transfer jika sertifikasi berbayar
            if ($sertifikasi->status === 'berbayar') {
                $request->validate([
                    'bukti_transfer' => 'required|image|mimes:jpg,jpeg,png|max:10048',
                ]);
            }

            // Proses upload bukti transfer jika berbayar
            $buktiTransferPath = null;
            if ($request->hasFile('bukti_transfer')) {
                $file = $request->file('bukti_transfer');
                $file_name = $file->hashName(); // nama unik
                $file->move(public_path('storage/assets/bukti-transfer'), $file_name);
                $buktiTransferPath = 'storage/assets/bukti-transfer/' . $file_name;
            }

            // Simpan data ke daftar_sertifikasis
            DaftarSertifikasi::create([
                'user_id'           => $user->id,
                'sertifikasi_id'    => $sertifikasi->id,
                'email'             => $user->email,
                'no_hp'             => $user->no_hp,
                'nama_lengkap'      => $profile->nama_lengkap,
                'asal_sekolah'      => $profile->asal_sekolah,
                'jurusan'           => $profile->jurusan_sekolah,
                'jenis_kelamin'     => $profile->jenis_kelamin,
                'tanggal_lahir'     => $profile->tanggal_lahir,
                'nik'               => $profile->nik,
                'tahun_kelulusan'   => $profile->tahun_kelulusan,
                'npsn'              => $profile->npsn,
                'provinsi'          => $profile->provinsi,
                'kota'              => $profile->kota,
                'alamat'            => $profile->alamat,
                'status_saat_ini'   => $profile->status_saat_ini,
                'bidang_pekerjaan'  => $profile->bidang_pekerjaan,
                'sertifikasi_terakhir' => $profile->sertifikasi_terakhir,
                'kesesuaian_sertifikasi' => $profile->kesesuaian_sertifikasi,
                'nama_universitas' => $profile->nama_universitas,
                'jurusan_universitas' => $profile->jurusan_universitas,
                'bukti_transfer'   => $buktiTransferPath,
            ]);

            return redirect()->back()->with('success', 'Pendaftaran sertifikasi berhasil.');
        } catch (\Exception $e) {
            return redirect()->route('alumni-siswa.sertifikasi')->with([
                'message' => 'Gagal mendaftar sertifikasi. Error: ' . $e->getMessage(),
                'alert-type' => 'danger',
            ]);
        }
    }


    public function storeDaftarLoker(Request $request, Loker $loker)
    {
        try {
            $user = Auth::user(); // user yang sedang login
            $profile = $user->alumniSiswaProfile;

            if (!$profile) {
                return redirect()->back()->with('error', 'Data profil alumni belum lengkap.');
            }

            // Cek jika sudah pernah mendaftar sertifikasi (opsional)
            $alreadyRegistered = DaftarLoker::where('user_id', $user->id)
                ->where('loker_id', $loker->id)
                ->exists();

            if ($alreadyRegistered) {
                return redirect()->back()->with([
                    'message' => 'Anda sudah pernah mendaftar lowongan kerja ini.',
                    'alert-type' => 'warning',
                ]);
            }

            $request->validate([
                'cv' => 'required|mimes:pdf|max:2048', // Max 2MB
            ]);
            // Upload CV
            $cv = $request->file('cv');
            $cv_name = $cv->hashName();
            $cv->move(public_path('storage/assets/cv'), $cv_name);

            // Simpan data ke daftar_sertifikasis
            DaftarLoker::create([
                'user_id'           => $user->id,
                'loker_id'          => $loker->id,
                'email'             => $user->email,
                'no_hp'             => $user->no_hp,
                'nama_lengkap'      => $profile->nama_lengkap,
                'asal_sekolah'      => $profile->asal_sekolah,
                'jurusan'           => $profile->jurusan_sekolah,
                'jenis_kelamin'     => $profile->jenis_kelamin,
                'tanggal_lahir'     => $profile->tanggal_lahir,
                'cv'                => $cv_name,
                'nik'               => $profile->nik,
                'tahun_kelulusan'   => $profile->tahun_kelulusan,
                'npsn'              => $profile->npsn,
                'provinsi'          => $profile->provinsi,
                'kota'              => $profile->kota,
                'alamat'            => $profile->alamat,
                'status_saat_ini'   => $profile->status_saat_ini,
                'bidang_pekerjaan'  => $profile->bidang_pekerjaan,
                'sertifikasi_terakhir' => $profile->sertifikasi_terakhir,
                'kesesuaian_sertifikasi' => $profile->kesesuaian_sertifikasi,
                'nama_universitas' => $profile->nama_universitas,
                'jurusan_universitas' => $profile->jurusan_universitas,
            ]);

            return redirect()->back()->with('success', 'Pendaftaran lowongan kerja berhasil.');
        } catch (\Exception $e) {
            return redirect()->route('alumni-siswa.loker')->with([
                'message' => 'Gagal mendaftar lowongan kerja. Error: ' . $e->getMessage(),
                'alert-type' => 'danger',
            ]);
        }
    }

    public function storeDaftarPelatihan(Request $request, Pelatihan $pelatihan)
    {
        try {
            $user = Auth::user(); // user yang sedang login
            $profile = $user->alumniSiswaProfile;

            if (!$profile) {
                return redirect()->back()->with('error', 'Data profil alumni belum lengkap.');
            }

            // Cek jika sudah pernah mendaftar sertifikasi (opsional)
            $alreadyRegistered = DaftarPelatihan::where('user_id', $user->id)
                ->where('pelatihan_id', $pelatihan->id)
                ->exists();

            if ($alreadyRegistered) {
                return redirect()->back()->with([
                    'message' => 'Anda sudah pernah mendaftar Pelatihan.',
                    'alert-type' => 'warning',
                ]);
            }

            // Validasi bukti transfer jika sertifikasi berbayar
            if ($pelatihan->status === 'berbayar') {
                $request->validate([
                    'bukti_transfer' => 'required|image|mimes:jpg,jpeg,png|max:10048',
                ]);
            }

            // Proses upload bukti transfer jika berbayar
            $buktiTransferPath = null;
            if ($request->hasFile('bukti_transfer')) {
                $file = $request->file('bukti_transfer');
                $file_name = $file->hashName(); // nama unik
                $file->move(public_path('storage/assets/bukti-transfer'), $file_name);
                $buktiTransferPath = 'storage/assets/bukti-transfer/' . $file_name;
            }


            // Simpan data ke daftar_sertifikasis
            DaftarPelatihan::create([
                'user_id'           => $user->id,
                'pelatihan_id'      => $pelatihan->id,
                'email'             => $user->email,
                'no_hp'             => $user->no_hp,
                'nama_lengkap'      => $profile->nama_lengkap,
                'asal_sekolah'      => $profile->asal_sekolah,
                'jurusan'           => $profile->jurusan_sekolah,
                'jenis_kelamin'     => $profile->jenis_kelamin,
                'tanggal_lahir'     => $profile->tanggal_lahir,
                'nik'               => $profile->nik,
                'tahun_kelulusan'   => $profile->tahun_kelulusan,
                'npsn'              => $profile->npsn,
                'provinsi'          => $profile->provinsi,
                'kota'              => $profile->kota,
                'alamat'            => $profile->alamat,
                'status_saat_ini'   => $profile->status_saat_ini,
                'bidang_pekerjaan'  => $profile->bidang_pekerjaan,
                'sertifikasi_terakhir' => $profile->sertifikasi_terakhir,
                'kesesuaian_sertifikasi' => $profile->kesesuaian_sertifikasi,
                'nama_universitas' => $profile->nama_universitas,
                'jurusan_universitas' => $profile->jurusan_universitas,
                'bukti_transfer'    => $buktiTransferPath,
            ]);

            return redirect()->back()->with('success', 'Pendaftaran pelatihan berhasil.');
        } catch (\Exception $e) {
            return redirect()->route('alumni-siswa.pelatihan')->with([
                'message' => 'Gagal mendaftar pelatihan. Error: ' . $e->getMessage(),
                'alert-type' => 'danger',
            ]);
        }
    }
}
